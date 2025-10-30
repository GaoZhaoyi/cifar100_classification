import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])


def load_data(data_dir, batch_size, dataset_type="CIFAR-10", pin_memory=True, use_augmentation=False):
    """
    加载数据并正确划分训练/验证集

    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        dataset_type: 数据集类型
        pin_memory: 是否使用锁页内存
        use_augmentation: 是否对训练集应用数据增强
    """
    # 1. 定义基础变换（无增强，用于验证集和测试集）
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # 2. 定义训练增强变换（仅用于训练集）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # 3. 加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=base_transform)

    # 4. 划分训练/验证集索引
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 5. 创建训练集和验证集子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # 6. 为训练集应用增强变换的包装器（仅当需要时）
    if use_augmentation:
        class TransformedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, index):
                x, y = self.subset[index]
                if self.transform:
                    # 将tensor转换回PIL图像以便应用增强
                    x = transforms.ToPILImage()(x)
                    x = self.transform(x)
                return x, y

            def __len__(self):
                return len(self.subset)

        # 训练集应用增强变换
        train_dataset = TransformedSubset(train_dataset, train_transform)
        # 验证集保持基础变换（无增强）

    # 7. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=False
    )

    is_augmented_data = "augmented" in data_dir

    print(f"Dataset loaded from: {data_dir}")
    print(f"Total images: {len(full_dataset)}")
    if is_augmented_data:
        print(f"Training set size: {len(train_dataset)} (with pre-augmentation)")
    elif use_augmentation:
        print(f"Training set size: {len(train_dataset)} (with runtime augmentation)")
    else:
        print(f"Training set size: {len(train_dataset)} (without augmentation)")
        print(f"Validation set size: {len(val_dataset)} (without augmentation)")
    print(f"Number of classes: {len(full_dataset.classes)}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float, dataset_type="CIFAR-10"):
    """
    Define the loss function and optimizer
    """
    criterion = LabelSmoothingLoss(smoothing=0.1)

    # 使用SGD优化器，适合CIFAR-100
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # 使用StepLR调度器，更快收敛
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, dataset_type="CIFAR-10"):
    """
    Train the model for one epoch
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
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
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


def test_epoch(model, dataloader, criterion, device, dataset_type="CIFAR-10"):
    """
    在测试集上评估模型（仅用于监控，不影响训练）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
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
    """
    with open(filename, 'w') as f:
        f.write(metrics)


def get_class_weights(dataset, num_classes):
    """
    Calculate class weights for imbalanced datasets
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
    """

    def f(epoch):
        if epoch < warmup_epochs:
            return warmup_factor * (epoch + 1) / warmup_epochs
        else:
            return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, f)
