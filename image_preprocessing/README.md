# Image Preprocessing for Myanmar Food Dataset

This script provides comprehensive image preprocessing for creating high-quality datasets for machine learning models, specifically designed for Myanmar food images.

## Features

### 1. **Standardized Image Sizing**
- Resizes all images to a consistent size (default: 224x224 pixels)
- Maintains aspect ratio with intelligent padding
- Prevents distortion of food images

### 2. **Image Enhancement**
- Automatic sharpness enhancement
- Contrast optimization
- Quality preservation

### 3. **Image Processing**
- **Resizing**: Standardized size with aspect ratio preservation
- **Enhancement**: Sharpness and contrast optimization
- **Quality Control**: Error handling and validation

### 4. **Quality Control**
- Error handling for corrupted images
- Detailed processing statistics
- Progress tracking
- Comprehensive logging

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from preprocess import ImagePreprocessor

# Initialize with default settings
preprocessor = ImagePreprocessor(
    input_dir="images/NanGyeeTote",
    output_dir="processed_images/NanGyeeTote",
    target_size=(224, 224),
    quality=95
)

# Process all images
stats = preprocessor.process_all_images()
```

### Custom Settings
```python
# Custom configuration
preprocessor = ImagePreprocessor(
    input_dir="your/input/path",
    output_dir="your/output/path",
    target_size=(512, 512),  # Larger size for high-resolution models
    quality=90  # Lower quality for smaller file sizes
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
├── original_image_processed.jpg      # Processed original
└── processing_stats.json            # Processing statistics
```

## Processing Statistics

The script generates detailed statistics including:
- Total images processed
- Success/failure rates
- Original image size statistics
- Processing errors
- Augmentation details

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Best Practices for Dataset Creation

### 1. **Image Quality**
- Use high-quality source images
- Ensure good lighting in original photos
- Capture food from multiple angles

### 2. **Dataset Diversity**
- Include various lighting conditions
- Capture different angles and distances
- Include seasonal variations

### 3. **Processing Strategy**
- The script processes each image once with enhancement
- Maintains original image quality while standardizing size
- Focuses on consistent, high-quality output

### 4. **Size Considerations**
- **224x224**: Good for most CNN models (ResNet, VGG, etc.)
- **512x512**: Better for detailed food recognition
- **1024x1024**: For high-resolution analysis

## Advanced Configuration

### Custom Processing Pipeline
```python
# Modify processing parameters
def enhance_image(self, image: np.ndarray) -> np.ndarray:
    # Custom enhancement logic
    pil_image = Image.fromarray(image)
    
    # Adjust sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)  # Increase sharpness
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)  # Increase contrast
    
    return np.array(pil_image)
```

### Batch Processing
```python
# Process multiple directories
directories = ["NanGyeeTote", "Mohinga", "TeaLeafSalad"]
for dir_name in directories:
    preprocessor = ImagePreprocessor(
        input_dir=f"images/{dir_name}",
        output_dir=f"processed_images/{dir_name}"
    )
    preprocessor.process_all_images()
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or image quality
2. **Slow Processing**: Use smaller target sizes
3. **Corrupted Images**: Check source image integrity
4. **Format Errors**: Ensure images are in supported formats

### Performance Tips

- Use SSD storage for faster I/O
- Process in smaller batches for large datasets
- Monitor memory usage during processing
- Use multiprocessing for large datasets

## Model Integration

The processed images are ready for:
- **CNN Models**: ResNet, VGG, EfficientNet
- **Transfer Learning**: Pre-trained models
- **Custom Architectures**: Any model requiring 224x224 input
- **Data Loaders**: PyTorch, TensorFlow, Keras

## Example Integration

```python
# PyTorch DataLoader example
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class FoodDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Usage
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = FoodDataset("processed_images/NanGyeeTote", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

This preprocessing pipeline ensures your Myanmar food images are optimized for machine learning model training with consistent quality and appropriate augmentations. 