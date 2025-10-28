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

def load_transforms():
    """
    Load the data transformations
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_data(data_dir, batch_size, dataset_type="CIFAR-10", pin_memory=True):
    """
    Load the data from the data directory and split it into training and validation sets

    Args:
        data_dir: The directory to load the data from
        batch_size: The batch size to use for the data loaders
        dataset_type: Type of dataset ("CIFAR-10" or "CIFAR-100")
        pin_memory: Whether to use pinned memory for data loading
    Returns:
        train_loader: The training data loader
        val_loader: The validation data loader
    """
    # Define data transformations: resize, convert to tensor, and normalize
    data_transforms = load_transforms()

    # Load the full dataset from the augmented data directory
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    # Split the dataset into training and validation sets (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator()
    )

    # 减少num_workers以降低内存使用
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # 从8减少到2
        pin_memory=pin_memory,
        persistent_workers=False  # 关闭持久化worker以减少内存
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # 从4减少到2
        pin_memory=pin_memory,
        persistent_workers=False
    )

    # Print dataset summary
    print(f"Dataset loaded from: {data_dir}")
    print(f"Total images: {len(full_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float, dataset_type="CIFAR-10"):
    """
    Define the loss function and optimizer
    This function is similar to the cell 3. Model Configuration in 04_model_training.ipynb
    Args:
        model: The model to train
        lr: Learning rate
        weight_decay: Weight decay
        dataset_type: Type of dataset ("CIFAR-10" or "CIFAR-100")
    Returns:
        criterion: The loss function
        optimizer: The optimizer
        scheduler: The scheduler
    """
    # criterion = nn.CrossEntropyLoss()

    criterion = LabelSmoothingLoss(smoothing=0.1)

    # Adjust optimizer settings for CIFAR-100 (more classes require different optimization)
    if dataset_type == "CIFAR-100":
        # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
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
