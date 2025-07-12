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
                 target_size: Tuple[int, int] = (1024, 1024),  # Fixed high-resolution size
                 quality: int = 100,  # Maximum quality
                 interpolation_method: str = "lanczos"):  # Best interpolation
        """
        Initialize the image preprocessor
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save processed images
            target_size: Fixed size for all images (width, height)
            quality: JPEG quality for saving (1-100)
            interpolation_method: Interpolation method for resizing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.quality = quality
        self.interpolation_method = interpolation_method
        
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
            'final_sizes': [],
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
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to fixed size with high quality and no blur"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Convert to PIL for better quality resizing
        pil_image = Image.fromarray(image)
        
        # Calculate aspect ratios
        img_ratio = w / h
        target_ratio = target_w / target_h
        
        if img_ratio > target_ratio:
            # Image is wider than target - crop width
            new_w = int(h * target_ratio)
            new_h = h
            left = (w - new_w) // 2
            top = 0
            right = left + new_w
            bottom = h
        else:
            # Image is taller than target - crop height
            new_w = w
            new_h = int(w / target_ratio)
            left = 0
            top = (h - new_h) // 2
            right = w
            bottom = top + new_h
        
        # Crop to maintain aspect ratio
        cropped = pil_image.crop((left, top, right, bottom))
        
        # Resize to target size using high-quality interpolation
        if self.interpolation_method == "lanczos":
            resized = cropped.resize(self.target_size, Image.Resampling.LANCZOS)
        elif self.interpolation_method == "bicubic":
            resized = cropped.resize(self.target_size, Image.Resampling.BICUBIC)
        else:
            resized = cropped.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return np.array(resized)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply high-quality image enhancements for sharp, clear images"""
        # Convert to PIL for enhancements
        pil_image = Image.fromarray(image)
        
        # Apply unsharp mask for better detail preservation
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=75, threshold=2))
        
        # Enhance sharpness moderately
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.15)
        
        # Enhance contrast slightly for better definition
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.08)
        
        # Enhance brightness slightly if needed
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.03)
        
        # Apply a second, very subtle unsharp mask for final sharpening
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=30, threshold=3))
        
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
            original_size = image.shape[:2]
            self.stats['original_sizes'].append(original_size)
            
            # Resize image intelligently
            resized = self.resize_image(image)
            
            # Store final size
            final_size = resized.shape[:2]
            self.stats['final_sizes'].append(final_size)
            
            # Enhance image
            enhanced = self.enhance_image(resized)
            
            # Create base filename
            base_name = image_path.stem
            
            # Save processed image
            output_path = self.output_dir / f"{base_name}_processed.jpg"
            if self.save_image(enhanced, output_path):
                self.stats['processed_images'] += 1
                logger.info(f"Saved: {output_path.name} ({final_size[1]}x{final_size[0]})")
            
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
            original_sizes = np.array(self.stats['original_sizes'])
            avg_orig_width = np.mean(original_sizes[:, 0])
            avg_orig_height = np.mean(original_sizes[:, 1])
            min_orig_width, max_orig_width = np.min(original_sizes[:, 0]), np.max(original_sizes[:, 0])
            min_orig_height, max_orig_height = np.min(original_sizes[:, 1]), np.max(original_sizes[:, 1])
        else:
            avg_orig_width = avg_orig_height = min_orig_width = max_orig_width = min_orig_height = max_orig_height = 0
        
        if self.stats['final_sizes']:
            final_sizes = np.array(self.stats['final_sizes'])
            avg_final_width = np.mean(final_sizes[:, 0])
            avg_final_height = np.mean(final_sizes[:, 1])
            min_final_width, max_final_width = np.min(final_sizes[:, 0]), np.max(final_sizes[:, 0])
            min_final_height, max_final_height = np.min(final_sizes[:, 1]), np.max(final_sizes[:, 1])
        else:
            avg_final_width = avg_final_height = min_final_width = max_final_width = min_final_height = max_final_height = 0
        
        stats_data = {
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'target_size': self.target_size,
            'interpolation_method': self.interpolation_method,
            'quality': self.quality,
            'statistics': {
                'total_images': self.stats['total_images'],
                'processed_images': self.stats['processed_images'],
                'failed_images': self.stats['failed_images'],
                'success_rate': (self.stats['processed_images'] / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0,
                'original_image_statistics': {
                    'average_width': float(avg_orig_width),
                    'average_height': float(avg_orig_height),
                    'min_width': int(min_orig_width),
                    'max_width': int(max_orig_width),
                    'min_height': int(min_orig_height),
                    'max_height': int(max_orig_height)
                },
                'final_image_statistics': {
                    'average_width': float(avg_final_width),
                    'average_height': float(avg_final_height),
                    'min_width': int(min_final_width),
                    'max_width': int(max_final_width),
                    'min_height': int(min_final_height),
                    'max_height': int(max_final_height)
                }
            },
            'errors': self.stats['processing_errors']
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Statistics saved to {stats_file}")

def main():
    """Main function to run the image preprocessing"""
    # Initialize preprocessor with fixed high-resolution settings
    preprocessor = ImagePreprocessor(
        input_dir="../images/NanGyeeTote",
        output_dir="../processed_images/NanGyeeTote",
        target_size=(1024, 1024),  # Fixed high-resolution size
        quality=100,  # Maximum quality
        interpolation_method="lanczos"  # Best interpolation for quality
    )
    
    # Process all images
    stats = preprocessor.process_all_images()
    
    # Print summary
    print("\n" + "="*50)
    print("FIXED-SIZE HIGH-RESOLUTION IMAGE PREPROCESSING")
    print("="*50)
    print(f"Total images found: {stats['total_images']}")
    print(f"Successfully processed: {stats['processed_images']}")
    print(f"Failed to process: {stats['failed_images']}")
    print(f"Success rate: {stats['processed_images']/stats['total_images']*100:.1f}%" if stats['total_images'] > 0 else "No images found")
    print(f"Output directory: {preprocessor.output_dir}")
    print(f"Fixed size: {preprocessor.target_size[0]}x{preprocessor.target_size[1]} pixels")
    print(f"Quality setting: {preprocessor.quality}")
    print(f"Interpolation: {preprocessor.interpolation_method}")
    print("="*50)

if __name__ == "__main__":
    main()
