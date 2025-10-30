import logging
import random
from pathlib import Path
from typing import List, Tuple

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
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        """
        Initialize the ImageAugmenter.
        Args:
            augmentations_per_image: Number of augmented versions per original image.
            seed: Random seed for reproducibility.
            save_original: Whether to save the original image with prefix 'orig_'.
            image_extensions: Tuple of valid image file extensions.
        """
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions

        self._set_seed()

        # Define Albumentations pipeline（简化并加强增强）
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

        Args:
            image: PIL Image to augment.

        Returns:
            Augmented PIL Image.
        """
        # Convert PIL to NumPy array (RGB)
        image_np = np.array(image)

        # Apply Albumentations transform
        augmented = self.transform(image=image_np)
        augmented_image_np = augmented["image"]

        # Convert back to PIL Image
        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        from pathlib import Path  # 确保导入Path库
        import os

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        total_augmented = 0  # 累计增强图像数量

        # 仅保留支持的图像后缀（避免非图像文件被误处理）
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [
            p for p in input_path.rglob('*')
            if p.suffix.lower() in valid_extensions and p.is_file()
        ]
        total_images = len(image_files)  # 总原始图像数量
        logger.info(f"Found {total_images} valid images to augment. Starting processing...")

        for idx, img_path in enumerate(image_files, 1):  # idx从1开始计数
            # 每1000张或最后一张时提示进度（与原逻辑一致）
            if idx % 1000 == 0 or idx == total_images:
                logger.info(
                    f"Progress: {idx}/{total_images} images processed, {total_augmented} augmented images generated")
            # 1. 加载原始图像（强制转为RGB，确保格式正确）
            try:
                # 用PIL打开并强制转为RGB（解决RGBA/灰度图问题）
                with Image.open(img_path) as img:
                    image = img.convert('RGB')  # 关键：统一转为RGB
            except Exception as e:
                logger.warning(f"Failed to load original image {img_path}: {e}")
                continue

            # 2. 构建输出路径（用Path自动处理Windows路径分隔符）
            rel_path = img_path.relative_to(input_path)  # 相对路径，保留类别结构
            target_dir = output_path / rel_path.parent  # 目标类别目录
            target_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

            # 3. 保存原始图像（若需要）
            if self.save_original:
                orig_name = f"orig_{rel_path.name}"
                orig_save_path = target_dir / orig_name
                try:
                    # 显式指定PNG格式，确保文件完整写入
                    with open(orig_save_path, 'wb') as f:
                        image.save(f, format='PNG')  # 强制PNG格式
                    # 验证保存结果
                    with Image.open(orig_save_path) as verify_img:
                        verify_img.load()  # 不仅verify，实际加载图像
                except Exception as e:
                    logger.error(f"Failed to save original image {orig_save_path}: {e}")
                    if orig_save_path.exists():
                        os.remove(orig_save_path)  # 删除损坏文件
                    continue

            # 4. 生成并保存增强图像
            for i in range(self.augmentations_per_image):
                try:
                    # 增强图像（确保输出是PIL图像）
                    augmented_img = self.augment_image(image.copy())
                    # 构建增强图像文件名
                    aug_name = f"aug_{i}_{rel_path.name}"
                    aug_save_path = target_dir / aug_name

                    # 保存增强图像（强制PNG格式）
                    with open(aug_save_path, 'wb') as f:
                        augmented_img.save(f, format='PNG')  # 显式格式

                    # 严格验证：加载图像并检查尺寸（确保非空）
                    with Image.open(aug_save_path) as verify_img:
                        verify_img.load()  # 实际加载像素
                        if verify_img.size != (32, 32):  # 确保尺寸正确（CIFAR图像32x32）
                            raise ValueError(f"Image size mismatch: {verify_img.size}, expected (32,32)")

                    total_augmented += 1
                except Exception as e:
                    logger.error(f"Failed to save augmented image {aug_save_path}: {e}")
                    if aug_save_path.exists():
                        os.remove(aug_save_path)  # 删除损坏文件
                    continue

        logger.info(
            f"Augmentation completed! Total processed images: {total_images}, Total augmented images generated: {total_augmented}. Output saved to: {output_dir}"
        )

    def _find_image_files(self, root: Path) -> List[Path]:
        """
        Recursively find all image files in directory.

        Args:
            root: Root directory path.

        Returns:
            List of image file paths.
        """
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5,
    seed: int = 42,
) -> None:
    """
    Backward-compatible wrapper for legacy code.

    Args:
        input_dir: Directory containing cleaned images (organized by class).
        output_dir: Directory to save augmented images.
        augmentations_per_image: Number of augmented versions per original image.
        seed: Random seed for reproducibility.
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image, seed=seed, save_original=True
    )
    augmenter.process_directory(input_dir, output_dir)
