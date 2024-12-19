import os
import argparse
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import cv2
import shutil 


def resize_image(image: np.array, target_resolution: int) -> np.array:
    height, width = image.shape[:2]
    if width <= height:
        height = target_resolution / width * height
        height = int((height / 8) + 1) * 8  # Ensure divisibility by 8
        width = target_resolution
    else:
        width = target_resolution / height * width
        width = int((width / 8) + 1) * 8  # Ensure divisibility by 8
        height = target_resolution

    print(f"Resized image from {image.shape[:2][::-1]} to {(width, height)}")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    

def get_args():
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl with captions for images in a directory.")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    target_resolution = 1024

    for file_name in os.listdir(args.input_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(args.input_dir, file_name)
            output_path = os.path.join(args.output_dir, file_name)
            image = Image.open(input_path).convert('RGB')
            width, height = image.size

            if min(width, height) > target_resolution:
                image = np.array(image)
                image = resize_image(image, target_resolution)
                image = Image.fromarray(image)
                image.save(output_path)
            else:
                sr_image = model.predict(image)
                sr_image = np.array(sr_image)
                sr_image = resize_image(sr_image, target_resolution)
                sr_image = Image.fromarray(sr_image)

                sr_image.save(output_path)


if __name__ == "__main__":
    main()