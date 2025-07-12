# Image Preprocessing for Myanmar Food Dataset

This script provides high-quality image preprocessing for creating standardized datasets for machine learning models, specifically designed for Myanmar food images with fixed high-resolution output.

## Features

### 1. **Fixed High-Resolution Processing**
- Resizes all images to a consistent high-resolution size (1024x1024 pixels)
- Maintains aspect ratio with intelligent center-cropping
- Prevents distortion while ensuring maximum detail preservation

### 2. **Maximum Quality Output**
- Uses maximum JPEG quality (100) for optimal image preservation
- Lanczos interpolation for superior resizing quality
- Advanced image enhancement pipeline for sharp, clear results

### 3. **Advanced Image Processing**
- **Intelligent Resizing**: Center-crop to maintain aspect ratio, then resize to fixed size
- **Multi-stage Enhancement**: Unsharp mask, sharpness, contrast, and brightness optimization
- **Quality Preservation**: Maximum quality settings throughout the pipeline

### 4. **Comprehensive Quality Control**
- Robust error handling for corrupted images
- Detailed processing statistics and progress tracking
- Comprehensive logging with timestamps
- JSON-based statistics export

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from preprocess import ImagePreprocessor

# Initialize with high-resolution settings
preprocessor = ImagePreprocessor(
    input_dir="images/NanGyeeTote",
    output_dir="processed_images/NanGyeeTote",
    target_size=(1024, 1024),  # Fixed high-resolution size
    quality=100,  # Maximum quality
    interpolation_method="lanczos"  # Best interpolation
)

# Process all images
stats = preprocessor.process_all_images()
```

### Custom Settings
```python
# Custom configuration for different resolutions
preprocessor = ImagePreprocessor(
    input_dir="your/input/path",
    output_dir="your/output/path",
    target_size=(512, 512),  # Medium resolution
    quality=95,  # High quality
    interpolation_method="bicubic"  # Alternative interpolation
)
```

### Command Line Usage
```bash
python preprocess.py
```

## Output Structure

The script creates the following structure:
```
processed_images/NanGyeeTote/
├── original_image_processed.jpg      # High-quality processed image
└── processing_stats.json            # Detailed processing statistics
```

## Processing Statistics

The script generates comprehensive statistics including:
- Total images processed
- Success/failure rates with percentages
- Original image size statistics (min, max, average)
- Final image size statistics
- Processing errors with detailed error messages
- Processing date and configuration details

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Processing Pipeline

### 1. **Image Loading**
- Robust loading with PIL for better format support
- Automatic RGB conversion for consistency
- Comprehensive error handling

### 2. **Intelligent Resizing**
- Center-crop to maintain aspect ratio
- Resize to fixed dimensions (1024x1024)
- High-quality Lanczos interpolation

### 3. **Multi-stage Enhancement**
- **Unsharp Mask**: Primary sharpening (radius=1.5, percent=75, threshold=2)
- **Sharpness Enhancement**: Moderate sharpness boost (1.15x)
- **Contrast Enhancement**: Subtle contrast improvement (1.08x)
- **Brightness Adjustment**: Slight brightness boost (1.03x)
- **Final Sharpening**: Secondary unsharp mask for detail preservation

### 4. **Quality Preservation**
- Maximum JPEG quality (100)
- Optimized saving with PIL
- Consistent color space handling

## Best Practices for High-Resolution Datasets

### 1. **Image Quality Requirements**
- Use high-quality source images (minimum 1024x1024 recommended)
- Ensure good lighting and focus in original photos
- Capture food from multiple angles for comprehensive datasets

### 2. **Dataset Considerations**
- **Storage**: High-resolution images require significant storage space
- **Processing Time**: Larger images take longer to process
- **Memory Usage**: Ensure sufficient RAM for batch processing

### 3. **Resolution Strategy**
- **1024x1024**: Optimal for detailed food analysis and modern AI models
- **512x512**: Good balance between quality and storage
- **224x224**: Traditional CNN size (not recommended for this pipeline)

## Advanced Configuration

### Custom Enhancement Pipeline
```python
def enhance_image(self, image: np.ndarray) -> np.ndarray:
    # Custom enhancement logic
    pil_image = Image.fromarray(image)
    
    # Custom unsharp mask settings
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2.0, percent=100, threshold=1))
    
    # Custom sharpness enhancement
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.3)  # More aggressive sharpening
    
    # Custom contrast enhancement
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.15)  # Higher contrast
    
    return np.array(pil_image)
```

### Batch Processing Multiple Directories
```python
# Process multiple food categories
directories = ["NanGyeeTote", "Mohinga", "TeaLeafSalad", "ShanNoodles"]
for dir_name in directories:
    preprocessor = ImagePreprocessor(
        input_dir=f"images/{dir_name}",
        output_dir=f"processed_images/{dir_name}",
        target_size=(1024, 1024),
        quality=100,
        interpolation_method="lanczos"
    )
    stats = preprocessor.process_all_images()
    print(f"Processed {dir_name}: {stats['processed_images']}/{stats['total_images']} successful")
```

## Performance Optimization

### Memory Management
- Process images one at a time to minimize memory usage
- Use SSD storage for faster I/O operations
- Monitor system resources during processing

### Processing Speed
- High-resolution processing is computationally intensive
- Consider processing during off-peak hours for large datasets
- Use multiprocessing for very large datasets (custom implementation needed)

## Troubleshooting

### Common Issues

1. **Memory Errors**: 
   - Reduce batch size or process images individually
   - Ensure sufficient RAM (8GB+ recommended for 1024x1024)
   - Close other applications during processing

2. **Slow Processing**: 
   - High-resolution processing is inherently slower
   - Use SSD storage for faster I/O
   - Process during low system usage periods

3. **Corrupted Images**: 
   - Check source image integrity
   - Verify supported file formats
   - Review error logs in processing_stats.json

4. **Quality Issues**: 
   - Ensure source images are high quality
   - Check that target_size is appropriate for your use case
   - Verify interpolation method settings

### Performance Tips

- **Storage**: Use SSD for both input and output directories
- **RAM**: Ensure 8GB+ available for 1024x1024 processing
- **CPU**: Multi-core systems will process faster
- **Batch Size**: Process in smaller batches for very large datasets

## Model Integration

The processed images are optimized for:
- **Modern AI Models**: Vision Transformers, ConvNeXt, EfficientNetV2
- **High-Resolution Analysis**: Detailed food texture and feature analysis
- **Transfer Learning**: Pre-trained models requiring 1024x1024 input
- **Custom Architectures**: Any model requiring high-resolution input

## Example Integration

```python
# PyTorch DataLoader for high-resolution images
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class HighResFoodDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('_processed.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Usage with high-resolution transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = HighResFoodDataset("processed_images/NanGyeeTote", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Smaller batch size for high-res
```

## Statistics Output Example

The script generates detailed JSON statistics:

```json
{
  "processing_date": "2024-01-15T10:30:00",
  "input_directory": "images/NanGyeeTote",
  "output_directory": "processed_images/NanGyeeTote",
  "target_size": [1024, 1024],
  "interpolation_method": "lanczos",
  "quality": 100,
  "statistics": {
    "total_images": 50,
    "processed_images": 48,
    "failed_images": 2,
    "success_rate": 96.0,
    "original_image_statistics": {
      "average_width": 1920.5,
      "average_height": 1440.3,
      "min_width": 800,
      "max_width": 4000,
      "min_height": 600,
      "max_height": 3000
    },
    "final_image_statistics": {
      "average_width": 1024,
      "average_height": 1024,
      "min_width": 1024,
      "max_width": 1024,
      "min_height": 1024,
      "max_height": 1024
    }
  },
  "errors": [
    {
      "file": "corrupted_image.jpg",
      "error": "Cannot identify image file"
    }
  ]
}
```

This preprocessing pipeline ensures your Myanmar food images are optimized for high-resolution machine learning model training with maximum quality preservation and consistent output standards. 