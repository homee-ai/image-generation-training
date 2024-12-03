import argparse
import cv2
import numpy as np
import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


class SegmentationModel:
    """
    A class used to represent a Segmentation Model for image processing.

    Methods
    -------
    __init__(device: str = "cuda")
        Initializes the SegmentationModel with the specified device.
    predict(image: np.array) -> np.array
        Predicts the segmentation map for the given image.
    colorize(seg_map: np.array) -> np.array
        Converts a segmentation map to a colorized image using the predefined palette.
    """

    palette = np.asarray(
        [
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ]
    )

    def __init__(self, device: str = "cuda"):
        """
        Initializes the segmentation model.
        Args:
            device (str): The device to run the model on, either "cuda" or "cpu". Default is "cuda".
        Attributes:
            device (str): The device to run the model on.
            processor (OneFormerProcessor): The processor for the segmentation model, loaded from pretrained weights.
            model (OneFormerForUniversalSegmentation): The segmentation model, loaded from pretrained weights and moved to the specified device.
        """

        self.device = device
        self.processor = OneFormerProcessor.from_pretrained(
            # "shi-labs/oneformer_ade20k_swin_tiny"
            "shi-labs/oneformer_ade20k_swin_large"
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            # "shi-labs/oneformer_ade20k_swin_tiny"
            "shi-labs/oneformer_ade20k_swin_large"
        ).to(self.device)

    def predict(self, image: np.array) -> np.array:
        """
        Predicts the segmentation map for a given input image.
        Args:
            image (np.array): The input image as a NumPy array.
        Returns:
            np.array: The predicted segmentation map as a NumPy array.
        """

        height, width = image.shape[:2]
        inputs = self.processor(image, ["semantic"], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]
        seg_map = seg_map.detach().cpu().numpy()

        return seg_map

    def colorize(self, seg_map: np.array) -> np.array:
        """
        Converts a segmentation map to a colorized image using a predefined palette.
        Args:
            seg_map (np.array): A 2D array representing the segmentation map where each value corresponds to a class index.
        Returns:
            np.array: A 3D array representing the colorized image with the same height and width as the input segmentation map,
                      and 3 color channels (RGB) with values in the range [0, 255].
        """

        seg_image = self.palette[seg_map].astype(np.uint8)
        return seg_image


def get_args():
    parser = argparse.ArgumentParser(description="Segmentation model inference")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input file path"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output file path"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = SegmentationModel(device="cuda")
    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg_map = model.predict(image)
    seg_image = model.colorize(seg_map)
    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, seg_image)


if __name__ == "__main__":
    main()
