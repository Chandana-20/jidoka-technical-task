import cv2 as cv
import numpy as np
import os
import json
import glob

def change_brightness_contrast(img: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    img_float = img.astype(np.float32)
    adjusted = np.clip((img_float * alpha) + beta, 0, 255)
    return adjusted.astype(np.uint8)


def replace_red_with_blue_hsv(img: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask1 = (hsv[:, :, 0] <= 10) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 20)
    mask2 = (hsv[:, :, 0] >= 170) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 20)
    red_mask = mask1 | mask2
    hsv[red_mask, 0] = 120
    result = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return result

def swap_image_halves(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    midpoint = width // 2
    left_half = img[:, :midpoint]
    right_half = img[:, midpoint:]
    return np.concatenate((right_half, left_half), axis=1)

def setup_directories(base_output: str) -> dict:
    subfolders = ["contrast_and_brightness", "convert_red", "swapped_halves", "combined"]
    paths = {}
    for sub in subfolders:
        folder_path = os.path.join(base_output, sub)
        paths[sub] = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    return paths

def main():
    
    with open("config.json", "r") as file:
        config = json.load(file)

    input_dir = config["paths"]["input_folder"]
    output_dir = config["paths"]["output_folder"]
    alpha = config["parameters"]["alpha"]
    beta = config["parameters"]["beta"]

    out_paths = setup_directories(output_dir)
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv.imread(img_path)

        img_bc = change_brightness_contrast(img, alpha, beta)
        cv.imwrite(os.path.join(out_paths["contrast_and_brightness"], f"bc_{img_name}"), img_bc)
        
        img_red = replace_red_with_blue_hsv(img)
        cv.imwrite(os.path.join(out_paths["convert_red"], f"red_{img_name}"), img_red)
        
        img_swap = swap_image_halves(img)
        cv.imwrite(os.path.join(out_paths["swapped_halves"], f"swap_{img_name}"), img_swap)
        
        combined = change_brightness_contrast(img, alpha, beta)
        combined = replace_red_with_blue_hsv(combined)
        combined = swap_image_halves(combined)
        
        cv.imwrite(os.path.join(out_paths["combined"], f"combined_{img_name}"), combined)

if __name__ == '__main__':
    main()