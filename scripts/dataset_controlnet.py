import os
import sys 
import json
import argparse
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join("..")))

from image_generation_training.models.segmentation_model import SegmentationModel


color_code_mapping = {
    1: "wall",
    2: "building",
    3: "sky",
    4: "floor",
    5: "tree",
    6: "ceiling",
    7: "road",
    8: "bed",
    9: "windowpane",
    10: "grass",
    11: "cabinet",
    12: "sidewalk",
    13: "person",
    14: "earth",
    15: "door",
    16: "table",
    17: "mountain",
    18: "plant",
    19: "curtain",
    20: "chair",
    21: "car",
    22: "water",
    23: "painting",
    24: "sofa",
    25: "shelf",
    26: "house",
    27: "sea",
    28: "mirror",
    29: "rug",
    30: "field",
    31: "armchair",
    32: "seat",
    33: "fence",
    34: "desk",
    35: "rock",
    36: "wardrobe",
    37: "lamp",
    38: "bathtub",
    39: "railing",
    40: "cushion",
    41: "base",
    42: "box",
    43: "column",
    44: "signboard",
    45: "chest of drawers",
    46: "counter",
    47: "sand",
    48: "sink",
    49: "skyscraper",
    50: "fireplace",
    51: "refrigerator",
    52: "grandstand",
    53: "path",
    54: "stairs",
    55: "runway",
    56: "case",
    57: "pool table",
    58: "pillow",
    59: "screen door",
    60: "stairway",
    61: "river",
    62: "bridge",
    63: "bookcase",
    64: "blind",
    65: "coffee table",
    66: "toilet",
    67: "flower",
    68: "book",
    69: "hill",
    70: "bench",
    71: "countertop",
    72: "stove",
    73: "palm",
    74: "kitchen island",
    75: "computer",
    76: "swivel chair",
    77: "boat",
    78: "bar",
    79: "arcade machine",
    80: "hovel",
    81: "bus",
    82: "towel",
    83: "light",
    84: "truck",
    85: "tower",
    86: "chandelier",
    87: "awning",
    88: "streetlight",
    89: "booth",
    90: "television",
    91: "airplane",
    92: "dirt track",
    93: "apparel",
    94: "pole",
    95: "land",
    96: "bannister",
    97: "escalator",
    98: "ottoman",
    99: "bottle",
    100: "buffet",
    101: "poster",
    102: "stage",
    103: "van",
    104: "ship",
    105: "fountain",
    106: "conveyer belt",
    107: "canopy",
    108: "washer",
    109: "plaything",
    110: "swimming pool",
    111: "stool",
    112: "barrel",
    113: "basket",
    114: "waterfall",
    115: "tent",
    116: "bag",
    117: "minibike",
    118: "cradle",
    119: "oven",
    120: "ball",
    121: "food",
    122: "step",
    123: "tank",
    124: "trade name",
    125: "microwave",
    126: "pot",
    127: "animal",
    128: "bicycle",
    129: "lake",
    130: "dishwasher",
    131: "screen",
    132: "blanket",
    133: "sculpture",
    134: "hood",
    135: "sconce",
    136: "vase",
    137: "traffic light",
    138: "tray",
    139: "ashcan",
    140: "fan",
    141: "pier",
    142: "crt screen",
    143: "plate",
    144: "monitor",
    145: "bulletin board",
    146: "shower",
    147: "radiator",
    148: "glass",
    149: "clock",
    150: "flag"
}


def generate_captions(input_dir):
    device = "cuda"
    # Initialize segmentation model
    seg_model = SegmentationModel(device)
    # Initialize the BLIP model and processor
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    oneformer_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_tiny"
        # "shi-labs/oneformer_ade20k_swin_large"
    )
    oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_tiny"
        # "shi-labs/oneformer_ade20k_swin_large"
    ).to(device)

    image_dir = os.path.join(input_dir, "image")
    output_dir = os.path.join(input_dir, "segmentation_image")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(input_dir, "metadata.jsonl")
    counter = 0
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for file_name in os.listdir(image_dir):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, file_name)
                    image = Image.open(image_path).convert("RGB")

                    # Generate segmentation image
                    np_image = np.array(image)
                    seg_image = seg_model.predict(np_image)
                    seg_image = seg_model.colorize(seg_image)
                    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
                    seg_image_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(seg_image_path, seg_image)
                    
                    # Generate caption
                    inputs = blip_processor(image, return_tensors="pt").to(device)
                    out = blip_model.generate(**inputs)
                    blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)
                    blip_caption = blip_caption.replace("\n", "")

                    inputs = oneformer_processor(images=image, task_inputs=["panoptic"], return_tensors="pt").to(device)
                    out = oneformer_model(**inputs)
                    panoptic_segmentation = oneformer_processor.post_process_panoptic_segmentation(out, target_sizes=[image.size[::-1]])[0]
                    segments_info = panoptic_segmentation["segments_info"]
                    detected_objects = {}
                    for segment in segments_info:
                        category_id = segment["label_id"] + 1
                        category_name = color_code_mapping[category_id]  # Map ID to category name
                        if category_name not in detected_objects:
                            detected_objects[category_name] = 0
                        detected_objects[category_name] += 1
                
                    # Print the list of objects found in the image
                    oneformer_caption = ""
                    for obj, count in detected_objects.items():
                        oneformer_caption += f"{obj}, "

                    # Create metadata entry
                    metadata = {
                        "image": os.path.join("image", file_name),
                        "conditioning_image": os.path.join("segmentation_image", file_name),
                        "text": f"{blip_caption},{oneformer_caption}"
                    }

                    # Write metadata to file
                    f.write(json.dumps(metadata) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl with captions for images in a directory.")
    parser.add_argument("--input_dir", type=str, help="Directory containing image files.")
    args = parser.parse_args()

    generate_captions(args.input_dir)