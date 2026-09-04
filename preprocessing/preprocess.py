import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, target_size=(256, 256), normalize=False):
        """
        Initializes the Image Preprocessor for classical ML feature extraction.
        
        Args:
            target_size (tuple): Target (width, height) to resize images. 
                                 Fixed size is crucial for consistent feature vectors (e.g., HOG).
            normalize (bool): Whether to scale pixel values to [0, 1].
                              Default is False. Classical texture extractors (LBP/GLCM) 
                              often work best with 8-bit integers [0, 255].
        """
        self.target_size = target_size
        self.normalize = normalize
        self.supported_formats = {'.bmp', '.png', '.jpg', '.jpeg'}
        
    def is_supported_file(self, file_path):
        """Checks if the file has a supported image extension (ignores .dat)."""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_formats
        
    def load_image(self, file_path):
        """Safely loads an image using OpenCV."""
        if not self.is_supported_file(file_path):
            return None
            
        try:
            # OpenCV loads in BGR by default
            img_bgr = cv2.imread(str(file_path))
            if img_bgr is None:
                print(f"Warning: Could not read image at {file_path}. File might be corrupted.")
                return None
            return img_bgr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
            
    def process_image(self, file_path):
        """
        Processes a single image file without modifying the original on disk.
        
        Returns:
            dict: Contains 'original_shape', 'bgr', 'rgb', and 'gray' resized images.
                  Returns None if processing fails.
        """
        img_bgr = self.load_image(file_path)
        if img_bgr is None:
            return None
            
        original_shape = img_bgr.shape
        
        # 1. Resize image
        # Using INTER_AREA which is generally preferred for decimation/downsampling.
        # Fixed size ensures feature descriptors have uniform lengths across the dataset.
        img_resized_bgr = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        
        # 2. Convert to RGB (useful for color-based features like color histograms)
        img_resized_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Convert to Grayscale (essential for texture/shape features like HOG, LBP, GLCM)
        img_resized_gray = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2GRAY)
        
        # 4. Optional normalization
        # We avoid aggressive preprocessing (like histogram equalization) at this stage 
        # to preserve the raw cell texture unless specifically needed later.
        if self.normalize:
            img_resized_rgb = img_resized_rgb.astype(np.float32) / 255.0
            img_resized_gray = img_resized_gray.astype(np.float32) / 255.0
            
        return {
            'original_shape': original_shape,
            'bgr': img_resized_bgr,
            'rgb': img_resized_rgb,
            'gray': img_resized_gray
        }

def run_tests():
    """
    Test section: Loads sample images, processes them, and saves visualizations.
    """
    data_dir = Path("data/archive")
    results_dir = Path("results/preprocessing")
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor with default 256x256 size
    preprocessor = ImagePreprocessor(target_size=(256, 256), normalize=False)
    
    if not data_dir.exists():
        print(f"Error: Dataset directory {data_dir} not found.")
        return

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(data_dir / d)]
    
    print("--- Running Preprocessing Tests ---")
    
    for cls in classes:
        # Some datasets are nested, e.g., data/archive/im_Dyskeratotic/im_Dyskeratotic
        cls_dir = data_dir / cls
        nested_dir = cls_dir / cls
        if nested_dir.exists() and nested_dir.is_dir():
            search_dir = nested_dir
        else:
            search_dir = cls_dir
            
        # Find all supported files in class dir
        files = [f for f in search_dir.iterdir() if preprocessor.is_supported_file(f)]
        
        if not files:
            print(f"No valid images found in {cls}")
            continue
            
        sample_file = files[0]
        print(f"\nProcessing sample from class: {cls}")
        print(f"File: {sample_file.name}")
        
        # Run preprocessing
        processed = preprocessor.process_image(sample_file)
        if processed is None:
            print("Failed to process image.")
            continue
            
        print(f"Original shape: {processed['original_shape']}")
        print(f"Resized RGB shape: {processed['rgb'].shape}")
        print(f"Resized Grayscale shape: {processed['gray'].shape}")
        
        # Save visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Load original directly for visualization purposes (convert BGR to RGB for plt)
        orig_img = cv2.cvtColor(cv2.imread(str(sample_file)), cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Original\n{processed['original_shape']}")
        axes[0].axis('off')
        
        axes[1].imshow(processed['rgb'])
        axes[1].set_title(f"Resized RGB\n{processed['rgb'].shape}")
        axes[1].axis('off')
        
        axes[2].imshow(processed['gray'], cmap='gray')
        axes[2].set_title(f"Resized Grayscale\n{processed['gray'].shape}")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = results_dir / f"test_{cls}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    run_tests()
