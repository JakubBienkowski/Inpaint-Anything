import cv2
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from stable_diffusion_inpaint import replace_img_with_sd
from utils import load_img_to_array, save_array_to_img

# Set your variables here
input_img_path = "./example/replace-anything/nike.jpg"
mask_img_path = "results/nike/mask_2.png"  # Path to your mask image
text_prompt = "Street photography of a sunny day in New York."
ng = " text, animals, people, objects"
output_dir = "./results"
dilate_kernel_size = None  # Set to an integer value if needed
seed = None  # Set to an integer value if needed
deterministic = False  # Set to True if needed

# Rest of your code remains the same
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Load input image and mask
img = load_img_to_array(input_img_path)
mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

# dilate mask to avoid unmasked edge effect
if dilate_kernel_size is not None:
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

# Create output directory
out_dir = Path(output_dir) / Path(input_img_path).stem
out_dir.mkdir(parents=True, exist_ok=True)

# Save the mask (optional)
mask_p = out_dir / "mask.png"
save_array_to_img(mask, mask_p)

# Inpaint with Stable Diffusion
if seed is not None:
    torch.manual_seed(seed)

img_replaced_p = out_dir / "replaced_image.png"
img_replaced = replace_img_with_sd(img, mask, text_prompt, ng, device=device)
save_array_to_img(img_replaced, img_replaced_p)