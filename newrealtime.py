
import os.path as osp
import numpy as np
import cv2


class CityscapesSingleImageProcessor:
    """
    CityscapesSingleImageProcessor processes a single image for inference.

    Args:
        image_path: The path to a single image file.
        mean: The mean values for normalization (default is (72.3924, 82.90902, 73.158325)).
    """

    def __init__(self, image_path='', mean=(72.3924, 82.90902, 73.158325)):
        self.image_path = image_path
        self.mean = mean

    def preprocess_image(self):
        # Load the image
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")

        image_name = osp.basename(self.image_path).split('.')[0]  # Get image name (without extension)
        size = image.shape  # Save original size

        image = np.asarray(image, np.float32)
        image -= self.mean  # Normalize the image
        image = image[:, :, ::-1]  # Convert BGR to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW
        image = np.ascontiguousarray(image)  # Ensure contiguous array to fix negative stride issue

        return image, np.array(size), image_name


def build_single_image_loader(image_path):
    """
    Builds a DataLoader-like function for processing a single image using CityscapesSingleImageProcessor.

    Args:
        image_path: Path to the single image to be processed.

    Returns:
        input_image: Processed image tensor with batch dimension.
        original_size: Original image size (H, W, C).
        image_name: Name of the image (without extension).
    """
    # Process the single image
    image_processor = CityscapesSingleImageProcessor(image_path=image_path)
    input_image, original_size, image_name = image_processor.preprocess_image()

    # Convert the image to a tensor and add a batch dimension
    input_image 

    return input_image, original_size, image_name