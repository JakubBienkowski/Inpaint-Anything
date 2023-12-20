import cv2
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import replace_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

# Set your variables here
input_img = "./example/replace-anything/nike.jpg"
coords_type = "key_in"
point_coords = [1000, 500]
point_labels = [1]
text_prompt = " street photography of a sunny day in New York."
ng = " text, animals, people, objects"
output_dir = "./results"
sam_model_type = "vit_h"
sam_ckpt = "./pretrained_models/sam_vit_h_4b8939.pth"
dilate_kernel_size = None  # Set to an integer value if needed
seed = None  # Set to an integer value if needed
deterministic = False  # Set to True if needed

# Rest of your code remains the same
device = "cuda" if torch.cuda.is_available() else "cpu"

if coords_type == "click":
    latest_coords = get_clicked_point(input_img)
elif coords_type == "key_in":
    latest_coords = point_coords
img = load_img_to_array(input_img)
height, width = img.shape[:2]
point_coords = [width // 2, height // 2]  # Center coordinates

masks, _, _ = predict_masks_with_sam(
    img,
    [point_coords],
    point_labels,
    model_type=sam_model_type,
    ckpt_p=sam_ckpt,
    device=device,
)
masks = masks.astype(np.uint8) * 255

# dilate mask to avoid unmasked edge effect
if dilate_kernel_size is not None:
    masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

# visualize the segmentation results
img_stem = Path(input_img).stem
out_dir = Path(output_dir) / img_stem
out_dir.mkdir(parents=True, exist_ok=True)
for idx, mask in enumerate(masks):
    # path to the results
    mask_p = out_dir / f"mask_{idx}.png"
    img_points_p = out_dir / f"with_points.png"
    img_mask_p = out_dir / f"with_{Path(mask_p).name}"

    # save the mask
    save_array_to_img(mask, mask_p)

    # save the pointed and masked image
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), [latest_coords], point_labels,
                size=(width*0.04)**2)
    plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
    show_mask(plt.gca(), mask, random_color=False)
    plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
    plt.close()

# fill the masked image
for idx, mask in enumerate(masks):
    if seed is not None:
        torch.manual_seed(seed)
    mask_p = out_dir / f"mask_{idx}.png"
    img_replaced_p = out_dir / f"replaced_with_{Path(mask_p).name}"
    img_replaced = replace_img_with_sd(
        img, mask, text_prompt, ng, device=device)
    save_array_to_img(img_replaced, img_replaced_p)
