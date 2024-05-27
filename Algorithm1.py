import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

sam_checkpoint = "./notebooks/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def scatter_points_proportional(image, baseline_density):

    points = []
    pixel_values = []


    num_zeros = np.sum(image == 0)
    num_ones = np.sum(image == 1)
    total_pixels = image.size
    
    if num_zeros == 0:  # 避免除以0
        density_0 = 0.2
    else:
        density_0 = baseline_density + (0.03 * (num_ones - num_zeros) / total_pixels)
    if num_ones == 0:
        density_255 = 0.2
    else:
        density_255 = baseline_density - (0.03 * (num_ones - num_zeros) / total_pixels)
    # 计算撒点的间距
    spacing_0 = int(1 / density_0) if density_0 != 0 else float('inf')
    spacing_255 = int(1 / density_255) if density_255 != 0 else float('inf')
    # 在图像上撒点
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 0 and y % spacing_0 == 0 and x % spacing_0 == 0:
                points.append((x, y))
                pixel_values.append(image[y, x])
            elif image[y, x] == 1 and y % spacing_255 == 0 and x % spacing_255 == 0:
                points.append((x, y))
                pixel_values.append(image[y, x])
    
    return points, pixel_values

def save_mask_only(mask, image_id, output_folder):
    plt.figure(figsize=(10, 10))
    show_mask(mask, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, image_id + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


def save_mask_on_image(image, masks, points, pixel_values, image_id, output_folder):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(points, pixel_values, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, image_id + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

def process_data(image_id):
    image = cv2.imread('./img/' + image_id + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_path = './mask/' + image_id + '.png'
    image_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    predictor.set_image(image)
    points, pixel_values = scatter_points_proportional(image_mask,0.07)
    np.set_printoptions(threshold=np.inf)
    pixel_values = [item // 1 for item in pixel_values]
    input_point = np.array(points)
    input_label = np.array(pixel_values)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    mask_input = logits[np.argmax(scores), :, :]


    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )


    # Save images to respective folders
    save_mask_only(masks, image_id, './mask_only')
    save_mask_on_image(image, masks, input_point, input_label, image_id, './mask_on_image')



image_extensions = ('.png', '.jpg', '.jpeg')
input_folder = './Images'
all_image_ids = [os.path.splitext(file_name)[0] for file_name in os.listdir(input_folder) if file_name.lower().endswith(image_extensions)]


for image_id in all_image_ids:
    try:
        process_data(image_id)
    except:
        with open('./wrong_images.txt', 'a') as f:
            f.write(f"{image_id}\n")