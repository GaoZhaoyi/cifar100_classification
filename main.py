#!/usr/bin/env python3
# cifar_pipeline.py - Complete pipeline for CIFAR-10/100 data preparation, augmentation, training and evaluation

import argparse
import logging
import os
import random
import numpy as np
import shutil

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torch.backends.cudnn as cudnn

# Import our custom modules
from scripts.data_download import download_and_extract_cifar10_data, download_and_extract_cifar100_data
from scripts.data_augmentation import augment_train_dataset
from scripts.model_architectures import create_model
from scripts.train_utils import (
    save_metrics,
    train_epoch,
    validate_epoch,
    test_epoch,
    save_checkpoint,
    define_loss_and_optimizer,
    EarlyStopping
)
from scripts.evaluation_metrics import (
    evaluate_model,
    top_k_accuracy,
    calculate_per_class_accuracy,
)

# 设置环境变量解决CuBLAS确定性警告
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cifar_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def optimize_gpu_settings():
    """Optimize GPU settings for faster training"""
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = False
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIFAR-10/100 Training Pipeline")

    # Dataset selection
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar100",
                        help="Dataset to use (cifar10 or cifar100)")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory for data storage")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")

    # Data augmentation
    parser.add_argument("--aug_count", type=int, default=3,
                        help="Number of augmentations per image")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128,  # 降低批次大小
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=200,  # 增加训练轮数
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1,  # 调整学习率
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 penalty)")

    # Mixup/CutMix参数
    parser.add_argument("--use_mixup", action="store_true",
                        help="Use Mixup data augmentation")
    parser.add_argument("--use_cutmix", action="store_true",
                        help="Use CutMix data augmentation")
    parser.add_argument("--mixup_prob", type=float, default=0.5,
                        help="Probability of applying Mixup")
    parser.add_argument("--cutmix_prob", type=float, default=0.5,
                        help="Probability of applying CutMix")

    # Model configuration
    parser.add_argument("--model_type", type=str, default="resnet18",
                        choices=["simple", "resnet18", "resnet34", "resnet50", "fast"],
                        help="Model architecture to use")

    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=50,  # 增加耐心
                        help="Early stopping patience")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Random seeds
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # 跳过步骤
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data collection and augmentation steps")
    parser.add_argument("--skip_collection", action="store_true",
                        help="Skip data collection step")
    parser.add_argument("--skip_augmentation", action="store_true",
                        help="Skip data augmentation step")

    return parser.parse_args()



def collect_data(args):
    """Collect data and pre-split into train/valid/test"""
    logger.info(f"Collecting {args.dataset} dataset...")

    print("Preparing data directory...")
    os.makedirs(args.data_dir + "/raw", exist_ok=True)
    print("Setup complete.")

    if args.dataset == "cifar10":
        train_dataset, test_dataset = download_and_extract_cifar10_data(
            root_dir=args.data_dir + "/raw",
            transform=None,
            save_images=True
        )
        num_classes = 10
    else:
        train_dataset, test_dataset = download_and_extract_cifar100_data(
            root_dir=args.data_dir + "/raw",
            transform=None,
            save_images=True
        )
        num_classes = 100

    # 预先划分训练数据为 train/valid
    raw_train_dir = args.data_dir + "/raw/train"
    split_train_dir = args.data_dir + "/raw/split/train"
    split_valid_dir = args.data_dir + "/raw/split/valid"

    if not os.path.exists(split_train_dir) or not os.path.exists(split_valid_dir):
        print("🚀 Pre-splitting training data into train/valid...")
        pre_split_train_data(raw_train_dir, split_train_dir, split_valid_dir, num_classes)
        print("✅ Pre-splitting completed!")
    else:
        print("✅ Train/valid data already pre-split")


def pre_split_train_data(raw_train_dir, split_train_dir, split_valid_dir, num_classes):
    """预先将训练数据划分为训练集和验证集"""
    # 获取所有类别
    class_names = os.listdir(raw_train_dir)

    # 创建目录结构
    for class_name in class_names:
        os.makedirs(os.path.join(split_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(split_valid_dir, class_name), exist_ok=True)

    # 对每个类别进行划分
    for class_name in class_names:
        class_dir = os.path.join(raw_train_dir, class_name)
        train_class_dir = os.path.join(split_train_dir, class_name)
        valid_class_dir = os.path.join(split_valid_dir, class_name)

        # 获取该类别的所有图像
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 随机打乱
        random.shuffle(image_files)

        # 90% 训练, 10% 验证 (对于CIFAR-100: 45000训练, 5000验证)
        split_idx = int(0.9 * len(image_files))
        train_files = image_files[:split_idx]
        valid_files = image_files[split_idx:]

        # 复制文件
        for file in train_files:
            shutil.copy2(os.path.join(class_dir, file), os.path.join(train_class_dir, file))
        for file in valid_files:
            shutil.copy2(os.path.join(class_dir, file), os.path.join(valid_class_dir, file))

        print(f"Class {class_name}: {len(train_files)} train, {len(valid_files)} valid")


def augment_data(args):
    """Prepare and augment pre-split training data only"""
    logger.info(f"Augmenting {args.dataset} training data...")

    # 原始划分的数据路径
    split_train_dir = args.data_dir + '/raw/split/train'  # 预划分的训练集
    split_valid_dir = args.data_dir + '/raw/split/valid'  # 预划分的验证集

    # 增强后数据路径
    augmented_train_dir = args.data_dir + '/augmented/train'

    # 检查原始数据目录
    if not os.path.exists(split_train_dir):
        print(f"❌ Error: Pre-split train data directory not found: {split_train_dir}")
        return False

    print(f"✅ Found pre-split train data at: {split_train_dir}")
    print(f"✅ Found pre-split valid data at: {split_valid_dir}")
    print(f"   Augmented train will be saved to: {augmented_train_dir}")
    print(f"   Valid data will be used directly from: {split_valid_dir}")
    print(f"   Number of augmentations per image: {args.aug_count}")

    # 强制生成增强数据（只处理训练集）
    if os.path.exists(augmented_train_dir):
        shutil.rmtree(augmented_train_dir)

    try:
        print("🚀 Starting training data augmentation...")
        # 只处理训练数据（应用增强）
        augment_train_dataset(
            train_input_dir=split_train_dir,
            train_output_dir=augmented_train_dir,
            augmentations_per_image=args.aug_count
        )

        print("\n🎉 Training data augmentation completed successfully!")

        # 验证生成的数据
        if os.path.exists(augmented_train_dir):
            train_count = sum([len(files) for r, d, files in os.walk(augmented_train_dir)])
            print(f"📊 Generated {train_count} augmented train images")
            print(f"📊 Valid data will be used directly from: {split_valid_dir}")
            return True
        else:
            print("❌ Error: Augmented train directory was not created")
            return False

    except Exception as e:
        print(f"❌ Error during training data augmentation: {e}")
        return False


def build_model(args):
    """Build the model"""
    if args.dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 100
    logger.info(f"Creating model with {num_classes} classes, {args.device} device...")
    model = create_model(
        num_classes=num_classes,
        device=args.device,
        model_type=args.model_type,
        pretrained=(args.model_type.startswith('resnet'))
    )
    return model


def train(args, model: nn.Module):
    # Determine dataset type for logging
    dataset_type = "CIFAR-10" if args.dataset == "cifar10" else "CIFAR-100"

    # 初始化Mixup和CutMix
    mixup = Mixup(alpha=0.2) if hasattr(args, 'use_mixup') and args.use_mixup else None
    cutmix = CutMix(alpha=1.0) if hasattr(args, 'use_cutmix') and args.use_cutmix else None

    # 设置Mixup/CutMix应用概率
    mixup_prob = getattr(args, 'mixup_prob', 0.3)
    cutmix_prob = getattr(args, 'cutmix_prob', 0.3)

    # Define loss and optimizer
    criterion, optimizer, scheduler = define_loss_and_optimizer(model, args.lr, args.weight_decay, dataset_type)

    # 调整学习率调度器
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # 初始化早停器（增加耐心）
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=0.001
    )

    # Initialize tracking variables
    best_val_loss = float("inf")
    best_val_acc = 0.0

    # Create directories
    os.makedirs(args.output_dir + "/models", exist_ok=True)
    os.makedirs(args.output_dir + "/results", exist_ok=True)

    print(
        f"Training configured for {args.num_epochs} epochs with early stopping patience of {args.early_stopping_patience}.")

    # 直接加载预处理好的数据（无运行时增强）
    augmented_train_path = args.data_dir + "/augmented/train"  # 增强的训练数据
    valid_path = args.data_dir + "/raw/split/valid"  # 原始验证数据（未增强）
    test_path = args.data_dir + "/raw/test"  # 原始测试数据（未增强）

    print(f"✅ Loading pre-processed AUGMENTED training data from: {augmented_train_path}")
    print(f"✅ Loading ORIGINAL validation data from: {valid_path}")
    print(f"✅ Loading ORIGINAL test data from: {test_path}")

    # 定义基础变换（无增强）
    from torchvision import transforms
    from scripts.train_utils import CIFAR10_MEAN, CIFAR10_STD
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # 加载训练数据（已增强）
    train_dataset = datasets.ImageFolder(root=augmented_train_path, transform=base_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    # 加载验证数据（原始数据，无增强）
    val_dataset = datasets.ImageFolder(root=valid_path, transform=base_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    # 加载测试数据用于监控（原始数据，无增强）
    test_dataset = datasets.ImageFolder(root=test_path, transform=base_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    print(f"Training set size: {len(train_dataset)} (with pre-augmentation)")
    print(f"Validation set size: {len(val_dataset)} (ORIGINAL, without augmentation)")
    print(f"Test set size: {len(test_dataset)} (ORIGINAL, without augmentation)")
    print(f"Number of classes: {len(train_dataset.classes)}")

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Train for one epoch with Mixup/CutMix
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, dataset_type,
            mixup=mixup, cutmix=cutmix, mixup_prob=mixup_prob, cutmix_prob=cutmix_prob
        )

        # Validate the model
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, args.device, dataset_type)

        # Update learning rate
        if hasattr(scheduler, 'step'):
            scheduler.step()

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 每10轮在测试集上评估一次（仅用于监控，不影响训练）
        if (epoch + 1) % 10 == 0:
            test_acc = test_epoch(model, test_loader, criterion, args.device, dataset_type)
            print(f"  🔍 Test Acc (monitoring only): {test_acc:.2f}%")

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

        # Check for improvement and save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                },
                args.output_dir + "/models/best_model.pth",
            )
            print("  ↳ Validation accuracy improved. Saving best model!")

    print("\nTraining completed!")

    # Load the best model checkpoint saved during training
    checkpoint = torch.load(args.output_dir + "/models/best_model.pth")
    model.load_state_dict(checkpoint["state_dict"])

    # Retrieve details from the checkpoint
    best_epoch = checkpoint["epoch"]
    best_val_acc_loaded = checkpoint["best_val_acc"]
    best_val_loss_loaded = checkpoint["best_val_loss"]

    print(
        f"Loaded best model from epoch {best_epoch} with validation accuracy {best_val_acc_loaded:.2f}% and loss {best_val_loss_loaded:.4f}")

    # Save the final model's state_dict
    torch.save(model.state_dict(), args.output_dir + "/models/final_model.pth")
    print(f"Final model state_dict saved to '{args.output_dir}/models/final_model.pth'.")

    return model, best_val_loss


def evaluate(args, model: nn.Module):
    """Evaluate the model on test data"""
    # Determine dataset type
    dataset_type = "CIFAR-10" if args.dataset == "cifar10" else "CIFAR-100"

    # Load the test dataset
    test_path = args.data_dir + "/raw/test"
    from torchvision import transforms
    from scripts.train_utils import CIFAR10_MEAN, CIFAR10_STD
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set the model to evaluation mode
    model.eval()

    # Define loss function
    criterion, _, _ = define_loss_and_optimizer(model, args.lr, args.weight_decay, dataset_type)

    # Evaluate the model
    test_loss, test_accuracy, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, criterion, args.device, dataset_type
    )

    print(f"\n{dataset_type} Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")

    # Calculate top-k accuracy for CIFAR-100
    if args.dataset == "cifar100":
        top1_acc = test_accuracy
        top5_acc = top_k_accuracy(all_labels, all_probs, k=5)
        print(f"  Top-5 Accuracy: {top5_acc:.2f}%")

    # Generate classification report
    metrics_str = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    print("\nClassification Report:")
    print(metrics_str)

    save_metrics(metrics_str, args.output_dir + "/classification_report.txt")

    # Calculate per-class accuracy
    per_class_acc = calculate_per_class_accuracy(all_labels, all_preds, test_dataset.classes)

    # Save evaluation results
    with open(args.output_dir + "/evaluation_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        if args.dataset == "cifar100":
            f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
        f.write("\nPer-class Accuracy:\n")
        for class_name, acc in per_class_acc.items():
            f.write(f"  {class_name}: {acc * 100:.2f}%\n")


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Set random seeds
    set_random_seeds(args.seed)

    # Optimize GPU settings
    optimize_gpu_settings()

    # Print configuration
    logger.info("Starting CIFAR pipeline with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # 如果不跳过数据准备，则执行数据收集和增强
    if not args.skip_data_prep:
        # 如果不跳过收集，则收集数据
        if not args.skip_collection:
            collect_data(args)

        # 如果不跳过增强，则进行数据增强
        if not args.skip_augmentation:
            augment_data(args)

    # Build model
    model = build_model(args)
    # Train
    train(args, model)
    # Evaluate
    evaluate(args, model)


if __name__ == "__main__":
    main()
