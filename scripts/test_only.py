#!/usr/bin/env python3
# test_only.py - Test existing model

import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

# Import our custom modules
from model_architectures import create_model
from train_utils import load_transforms
from evaluation_metrics import evaluate_model, top_k_accuracy, calculate_per_class_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Existing CIFAR-10/100 Model")

    # Dataset selection
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar100",
                        help="Dataset to use (cifar10 or cifar100)")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory for data storage")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory where models are saved")

    # Model configuration
    parser.add_argument("--model_type", type=str, default="resnet50",
                        choices=["simple", "resnet18", "resnet34", "resnet50"],
                        help="Model architecture to use")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for evaluation")

    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to model checkpoint (default: use best_model.pth)")

    return parser.parse_args()


def load_model_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully!")
    return model


def evaluate(args, model: nn.Module):
    """Evaluate the model on test data"""
    # Determine dataset type
    dataset_type = "CIFAR-10" if args.dataset == "cifar10" else "CIFAR-100"

    # Load the test dataset from the specified directory
    test_data_dir = args.data_dir + "/raw/test"
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=load_transforms())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Set the model to evaluation mode
    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    print(f"Evaluating {dataset_type} model...")
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

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "final_test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        if args.dataset == "cifar100":
            f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")

    print(f"Results saved to {args.output_dir}/final_test_results.txt")


def main():
    """Main function"""
    args = parse_args()

    # Print configuration
    logger.info("Testing model with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Build model
    if args.dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 100

    model = create_model(num_classes=num_classes, device=args.device, model_type=args.model_type)

    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.output_dir, "models", "best_model.pth")

    model = load_model_checkpoint(model, checkpoint_path, args.device)

    # Evaluate
    evaluate(args, model)


if __name__ == "__main__":
    main()
