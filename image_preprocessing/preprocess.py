import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, 
                 input_dir: str = "images/NanGyeeTote",
                 output_dir: str = "processed_images/NanGyeeTote",
                 target_size: Tuple[int, int] = (224, 224),
                 quality: int = 95):
        """
        Initialize the image preprocessor
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save processed images
            target_size: Target size for all images (width, height)
            quality: JPEG quality for saving (1-100)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.quality = quality
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'original_sizes': [],
            'processing_errors': []
        }
        
        # No augmentation pipeline needed
        
    def _create_augmentation_pipeline(self):
        """Augmentation removed - keeping method for compatibility"""
        pass
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image with error handling"""
        try:
            # Try PIL first for better format support
            pil_image = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            self.stats['processing_errors'].append({
                'file': str(image_path),
                'error': str(e)
            })
            return None
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image maintaining aspect ratio with padding"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor to fit image within target size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply basic image enhancements"""
        # Convert to PIL for enhancements
        pil_image = Image.fromarray(image)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return np.array(pil_image)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert normalized image back to uint8"""
        return (image * 255).astype(np.uint8)
    
    def apply_augmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Augmentation removed - return only original image"""
        return [image]  # Return only the original image
    
    def save_image(self, image: np.ndarray, output_path: Path, suffix: str = "") -> bool:
        """Save image with error handling"""
        try:
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = self.denormalize_image(image)
            
            # Convert to PIL for better saving
            pil_image = Image.fromarray(image)
            
            # Save with specified quality
            pil_image.save(output_path, 'JPEG', quality=self.quality, optimize=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            return False
    
    def process_single_image(self, image_path: Path) -> bool:
        """Process a single image through the complete pipeline"""
        try:
            # Load image
            image = self.load_image(image_path)
            if image is None:
                self.stats['failed_images'] += 1
                return False
            
            # Store original size
            self.stats['original_sizes'].append(image.shape[:2])
            
            # Resize image
            resized = self.resize_image(image, self.target_size)
            
            # Enhance image
            enhanced = self.enhance_image(resized)
            
            # Create base filename
            base_name = image_path.stem
            
            # Save original processed image
            output_path = self.output_dir / f"{base_name}_processed.jpg"
            if self.save_image(enhanced, output_path):
                self.stats['processed_images'] += 1
            
            # No augmentation - only save the processed original
            # augmented_images = self.apply_augmentation(enhanced)
            # for i, aug_image in enumerate(augmented_images[1:], 1):  # Skip first (original)
            #     aug_output_path = self.output_dir / f"{base_name}_aug_{i}.jpg"
            #     if self.save_image(aug_image, aug_output_path):
            #         self.stats['processed_images'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            self.stats['failed_images'] += 1
            self.stats['processing_errors'].append({
                'file': str(image_path),
                'error': str(e)
            })
            return False
    
    def get_image_files(self) -> List[Path]:
        """Get all supported image files from input directory"""
        image_files = []
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        return sorted(image_files)
    
    def process_all_images(self) -> Dict:
        """Process all images in the input directory"""
        logger.info(f"Starting image preprocessing...")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target size: {self.target_size}")
        
        # Get all image files
        image_files = self.get_image_files()
        self.stats['total_images'] = len(image_files)
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
            self.process_single_image(image_path)
        
        # Generate and save statistics
        self._save_statistics()
        
        logger.info("Image preprocessing completed!")
        logger.info(f"Processed: {self.stats['processed_images']}")
        logger.info(f"Failed: {self.stats['failed_images']}")
        
        return self.stats
    
    def _save_statistics(self):
        """Save processing statistics to JSON file"""
        stats_file = self.output_dir / "processing_stats.json"
        
        # Calculate additional statistics
        if self.stats['original_sizes']:
            sizes = np.array(self.stats['original_sizes'])
            avg_width = np.mean(sizes[:, 0])
            avg_height = np.mean(sizes[:, 1])
            min_width, max_width = np.min(sizes[:, 0]), np.max(sizes[:, 0])
            min_height, max_height = np.min(sizes[:, 1]), np.max(sizes[:, 1])
        else:
            avg_width = avg_height = min_width = max_width = min_height = max_height = 0
        
        stats_data = {
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'target_size': self.target_size,
            'quality': self.quality,
            'statistics': {
                'total_images': self.stats['total_images'],
                'processed_images': self.stats['processed_images'],
                'failed_images': self.stats['failed_images'],
                'success_rate': (self.stats['processed_images'] / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0,
                'original_image_statistics': {
                    'average_width': float(avg_width),
                    'average_height': float(avg_height),
                    'min_width': int(min_width),
                    'max_width': int(max_width),
                    'min_height': int(min_height),
                    'max_height': int(max_height)
                }
            },
            'errors': self.stats['processing_errors']
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Statistics saved to {stats_file}")

def main():
    """Main function to run the image preprocessing"""
    # Initialize preprocessor with custom settings
    preprocessor = ImagePreprocessor(
        input_dir="../images/NanGyeeTote",
        output_dir="../processed_images/NanGyeeTote",
        target_size=(224, 224),  # Standard size for many ML models
        quality=95  # High quality for dataset
    )
    
    # Process all images
    stats = preprocessor.process_all_images()
    
    # Print summary
    print("\n" + "="*50)
    print("IMAGE PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Total images found: {stats['total_images']}")
    print(f"Successfully processed: {stats['processed_images']}")
    print(f"Failed to process: {stats['failed_images']}")
    print(f"Success rate: {stats['processed_images']/stats['total_images']*100:.1f}%" if stats['total_images'] > 0 else "No images found")
    print(f"Output directory: {preprocessor.output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
