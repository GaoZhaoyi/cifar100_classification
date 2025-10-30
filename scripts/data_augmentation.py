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

        # Define Albumentations pipeline（新增RandomCrop和Cutout）
        # self.transform = A.Compose(
        #     [
        #         A.RandomCrop(height=28, width=28, pad_if_needed=True, p=0.8),  # 随机裁剪后自动填充回32x32
        #         A.Resize(height=32, width=32),
        #         A.Rotate(limit=15, p=0.8),
        #         A.HorizontalFlip(p=0.5),
        #         A.ShiftScaleRotate(
        #             shift_limit=0.1,
        #             scale_limit=0.1,
        #             rotate_limit=0,
        #             p=0.8,
        #             border_mode=0,  # cv2.BORDER_CONSTANT
        #         ),
        #         A.ColorJitter(
        #             brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
        #         ),
        #         A.OneOf(
        #             [
        #                 A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        #                 A.MotionBlur(blur_limit=7, p=0.5),
        #             ],
        #             p=0.3,
        #         ),
        #         A.RandomBrightnessContrast(p=0.2),
        #     ]
        # )

        self.transform = A.Compose(
            [
                # 1. 修正RandomCrop：2.0+版本用`pad_if_needed=True`+`border_mode`替代`padding`
                A.RandomCrop(
                    height=32,
                    width=32,
                    pad_if_needed=True,  # 自动填充（替代原padding=4）
                    border_mode=cv2.BORDER_CONSTANT,  # 填充模式（黑色填充）
                    p=0.9
                ),
                # 2. 保留Rotate：参数不变（2.0+版本兼容）
                A.Rotate(limit=20, p=0.9),
                # 3. 保留HorizontalFlip：参数不变
                A.HorizontalFlip(p=0.5),
                # 4. 修正ShiftScaleRotate警告：用Affine替代（功能完全一致，消除警告）
                A.Affine(
                    translate_percent={"x": 0.1, "y": 0.1},  # 对应原shift_limit=0.1
                    scale=(0.9, 1.1),  # 对应原scale_limit=0.1（0.9~1.1倍缩放）
                    rotate=0,  # 对应原rotate_limit=0（不旋转，与Rotate模块分工）
                    p=0.8,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                # 5. 保留ColorJitter：参数不变（兼容2.0+）
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.15,
                    p=0.9
                ),
                # 6. 修正GaussNoise：2.0+版本用`var_limit`的新写法（或改用RandomGamma，避免参数警告）
                # 这里改用RandomGamma（效果类似，且无参数警告，更稳定）
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # 伽马校正：模拟亮度噪声，替代GaussNoise
                # 7. 保留GaussianBlur+MotionBlur：参数不变
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.6),
                        A.MotionBlur(blur_limit=7, p=0.5),
                    ],
                    p=0.5
                ),
                # 8. 修正CoarseDropout：2.0+版本参数名调整（holes→num_holes，size→max_size）
                A.CoarseDropout(
                    # 1. 遮挡数量：用 num_holes_range 指定范围（2-4个，替代旧的 max_holes）
                    num_holes_range=(2, 4),
                    # 2. 遮挡高度：用 hole_height_range 指定范围（8-10像素，固定像素值需传int元组）
                    hole_height_range=(8, 10),
                    # 3. 遮挡宽度：与高度保持一致，形成正方形遮挡
                    hole_width_range=(8, 10),
                    # 4. 填充颜色：用 fill 替代 fill_value，0表示黑色（RGB通用）
                    fill=0,
                    # 5. 触发概率：30%，与你的需求一致
                    p=0.3
                ),

                # 9. 修正RandomShear：2.0+版本无RandomShear，用Affine的shear参数替代
                A.Affine(
                    shear={"x": 15, "y": 15},  # 剪切角度±15°（x/y分别控制水平/垂直剪切）
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                # 10. 保留RandomBrightnessContrast：参数不变
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
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