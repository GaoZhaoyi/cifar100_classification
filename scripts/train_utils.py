import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler
import torch.cuda.amp as amp

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EarlyStopping:
    """标准化早停机制（防止过拟合，验证损失连续不下降则停止）"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience  # 容忍次数
        self.min_delta = min_delta  # 最小改进阈值
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def load_transforms(is_train: bool = True):
    """
    Load the data transformations（区分训练/测试集，修正水平翻转类名）
    Args:
        is_train: True for training transforms (with augmentation), False for test
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),  # 修正：HorizontalFlip → RandomHorizontalFlip
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])


def load_data(data_dir, batch_size, dataset_type="CIFAR-10", pin_memory=True):
    """
    修正：先划分原始数据为训练/验证集，再仅对训练集增强，避免数据泄露
    """
    # 1. 加载原始数据集（不应用增强，仅基础变换）
    base_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=base_transform)

    # 2. 划分训练/验证集索引（基于原始数据，无增强）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定种子确保划分稳定
    )

    # 3. 对训练集应用增强，验证集保持原始变换
    train_transform = load_transforms(is_train=True)  # 含增强的训练变换
    val_transform = load_transforms(is_train=False)    # 无增强的验证变换

    # 4. 创建带不同变换的训练/验证子集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataset.dataset.transform = train_transform  # 训练集覆盖为增强变换

    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = val_transform      # 验证集保持无增强

    # 5. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=False
    )

    # 打印数据集信息
    print(f"Dataset loaded from: {data_dir}")
    print(f"Total原始图像数: {len(full_dataset)}")
    print(f"训练集大小（含增强）: {len(train_dataset)}")
    print(f"验证集大小（无增强）: {len(val_dataset)}")
    print(f"类别数: {len(full_dataset.classes)}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float, dataset_type="CIFAR-10"):
    """
    Define the loss function and optimizer
    This function is similar to the cell 3. Model Configuration in 04_model_training.ipynb
    Args:
        model: The model to train
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        dataset_type: Type of dataset ("CIFAR-10" or "CIFAR-100")
    Returns:
        criterion: The loss function
        optimizer: The optimizer
        scheduler: The scheduler
    """
    criterion = LabelSmoothingLoss(smoothing=0.1)  # 标签平滑（已用，继续保留）

    # Adjust optimizer settings for CIFAR-100 (more classes require different optimization)
    if dataset_type == "CIFAR-100":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-7)     # T_max 需要设置为训练轮数
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    return criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, dataset_type="CIFAR-10"):
    """
    Train the model for one epoch
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        dataset_type: Type of dataset ("CIFAR-10" or "CIFAR-100")
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 添加混合精度缩放器
    scaler = GradScaler()

    progress_bar = tqdm(dataloader, desc=f"Training ({dataset_type})", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # 使用混合精度训练
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, dataset_type="CIFAR-10"):
    """
    Validate the model
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        dataset_type: Type of dataset ("CIFAR-10" or "CIFAR-100")
    Returns:
        Average loss and accuracy for the validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation ({dataset_type})", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)


def get_class_weights(dataset, num_classes):
    """
    Calculate class weights for imbalanced datasets
    Args:
        dataset: The dataset
        num_classes: Number of classes
    Returns:
        Class weights tensor
    """
    # Count samples per class
    class_counts = [0] * num_classes
    for _, label in dataset:
        class_counts[label] += 1

    # Calculate weights (inverse frequency)
    total_samples = len(dataset)
    class_weights = [total_samples / (num_classes * count) for count in class_counts]

    return torch.FloatTensor(class_weights)


def warmup_lr_scheduler(optimizer, warmup_epochs, warmup_factor):
    """
    Create a warmup learning rate scheduler
    Args:
        optimizer: The optimizer
        warmup_epochs: Number of warmup epochs
        warmup_factor: Warmup factor
    """

    def f(epoch):
        if epoch < warmup_epochs:
            return warmup_factor * (epoch + 1) / warmup_epochs
        else:
            return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, f)