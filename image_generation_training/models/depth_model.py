import argparse
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthModel:
    """
    A class used to represent a Depth Estimation Model.

    Methods
    -------
    predict(image: np.array) -> np.array
        Predicts the depth map for a given image.
    normalize(depth: np.array) -> np.array
        Normalizes the depth map to a range of 0 to 255.
    """

    def __init__(self, device: str = "cuda"):
        """
        Constructs all the necessary attributes for the DepthModel object.

        Parameters
        ----------
        device : str, optional
            The device to run the model on (default is "cuda").
        """
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).to(self.device)

    def predict(self, image: np.array) -> np.array:
        """
        Predicts the depth map for a given image.

        Parameters
        ----------
        image : np.array
            The input image for which the depth map is to be predicted.

        Returns
        -------
        np.array
            The predicted depth map.
        """
        height, width = image.shape[:2]
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(height, width),
            mode="nearest",
        )
        depth = depth.detach().squeeze().cpu().numpy()

        return depth

    def normalize(self, depth: np.array) -> np.array:
        """
        Normalizes the depth map to a range of 0 to 255.

        Parameters
        ----------
        depth : np.array
            The input depth map to be normalized.

        Returns
        -------
        np.array
            The normalized depth map as an 8-bit image.
        """
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth_image = (depth * 255).astype(np.uint8)
        return depth_image


def get_args():
    parser = argparse.ArgumentParser(description="Depth model inference")
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
    model = DepthModel(device="cuda")
    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = model.predict(image)
    cv2.imwrite(args.output, depth_map)


if __name__ == "__main__":
    main()
