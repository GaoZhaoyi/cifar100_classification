import logging
import random
from pathlib import Path
from typing import List, Tuple
import shutil

import numpy as np
import torch
import albumentations as A
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    """Class to handle image augmentation operations using Albumentations."""

    def __init__(
            self,
            augmentations_per_image: int = 5,
            seed: int = 42,
            image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        """
        Initialize the ImageAugmenter.
        """
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.image_extensions = image_extensions

        self._set_seed()

        # Define Albumentations pipeline
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.6),
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(4, 8),
                    hole_width_range=(4, 8),
                    fill=0,
                    p=0.5
                ),
                A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
            ]
        )

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation transforms to a single image using Albumentations.
        """
        # Convert PIL to NumPy array (RGB)
        image_np = np.array(image)

        # Apply Albumentations transform
        augmented = self.transform(image=image_np)
        augmented_image_np = augmented["image"]

        # Convert back to PIL Image
        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def process_train_directory(self, input_dir: str, output_dir: str) -> None:
        """处理训练数据目录 - 应用增强"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        total_augmented = 0

        # 仅保留支持的图像后缀
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [
            p for p in input_path.rglob('*')
            if p.suffix.lower() in valid_extensions and p.is_file()
        ]
        total_images = len(image_files)
        logger.info(f"Found {total_images} train images to augment. Starting processing...")

        for idx, img_path in enumerate(image_files, 1):
            if idx % 1000 == 0 or idx == total_images:
                logger.info(
                    f"Progress: {idx}/{total_images} images processed, {total_augmented} augmented images generated")

            try:
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load original image {img_path}: {e}")
                continue

            rel_path = img_path.relative_to(input_path)
            target_dir = output_path / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)

            # 生成并保存增强图像
            for i in range(self.augmentations_per_image):
                try:
                    augmented_img = self.augment_image(image.copy())
                    aug_name = f"aug_{i}_{rel_path.name}"
                    aug_save_path = target_dir / aug_name

                    with open(aug_save_path, 'wb') as f:
                        augmented_img.save(f, format='PNG')

                    with Image.open(aug_save_path) as verify_img:
                        verify_img.load()
                        if verify_img.size != (32, 32):
                            raise ValueError(f"Image size mismatch: {verify_img.size}, expected (32,32)")

                    total_augmented += 1
                except Exception as e:
                    logger.error(f"Failed to save augmented image {aug_save_path}: {e}")
                    if aug_save_path.exists():
                        os.remove(aug_save_path)
                    continue

        logger.info(
            f"Train data augmentation completed! Generated {total_augmented} augmented images. Output saved to: {output_dir}"
        )

    def _find_image_files(self, root: Path) -> List[Path]:
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_train_dataset(
        train_input_dir: str,
        train_output_dir: str,
        augmentations_per_image: int = 5,
        seed: int = 42,
) -> None:
    """
    对训练数据进行增强处理

    Args:
        train_input_dir: 原始训练数据目录
        train_output_dir: 增强后的训练数据输出目录
        augmentations_per_image: 每张图像的增强数量
        seed: 随机种子
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image, seed=seed
    )

    # 处理训练数据（应用增强）
    augmenter.process_train_directory(train_input_dir, train_output_dir)
