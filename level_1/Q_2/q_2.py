import cv2 as cv
import numpy as np
import os

def blur_levels(img):
    blur_level_1 = cv.GaussianBlur(img, (5, 5), 0)
    blur_level_2 = cv.GaussianBlur(img, (25, 25), 0)
    blur_level_3 = cv.GaussianBlur(img, (101, 101), 0)
    
    return blur_level_1, blur_level_2, blur_level_3

def blur_methods(img):
    blur_median = cv.medianBlur(img, 9)
    blur_bilateral = cv.bilateralFilter(img, 9, 75, 75)
    
    return blur_median, blur_bilateral

def sharp_levels_unsharp(img, amount, sigma):
    blurred = cv.GaussianBlur(img, (0, 0), sigma)
    sharp = cv.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

    return sharp

def sharp_levels_kernel(img):
    kernels = [
        np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]], dtype=np.float32),

        np.array([[-1, -1, -1],
                  [-1, 9, -1],
                  [-1, -1, -1]], dtype=np.float32),

        np.array([[-1, -2, -1],
                  [-2, 13, -2],
                  [-1, -2, -1]], dtype=np.float32)
    ]

    results = []
    for k in kernels:
        results.append(cv.filter2D(img, -1, k))
        
    return tuple(results)


def main():
    input_path = r"D:\Chandana\Technical_task\level_1\input_images\input_3.jpg"
    img = cv.imread(input_path)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    blurred = os.path.join(output_dir, "blurred")
    sharpened = os.path.join(output_dir, "sharpened")
    
    os.makedirs(blurred, exist_ok=True)
    os.makedirs(sharpened, exist_ok=True)
    
    blur_level_1, blur_level_2, blur_level_3 = blur_levels(img)
    blur_median, blur_bilateral = blur_methods(img)
    
    cv.imwrite(os.path.join(blurred, "blur_level_1.jpg"), blur_level_1)
    cv.imwrite(os.path.join(blurred, "blur_level_2.jpg"), blur_level_2)
    cv.imwrite(os.path.join(blurred, "blur_level_3.jpg"), blur_level_3)
    
    cv.imwrite(os.path.join(blurred, "blur_median.jpg"), blur_median)
    cv.imwrite(os.path.join(blurred, "blur_bilateral.jpg"), blur_bilateral)
    
    unsharp_configs = [
        (1.0, 1.0),
        (1.5, 1.5),
        (2.0, 2.0)
    ]
    
    for i, (amount, sigma) in enumerate(unsharp_configs, 1):
        sharp = sharp_levels_unsharp(img, amount, sigma)
        cv.imwrite(os.path.join(output_dir, f"unsharp_level_{i}.jpg"), sharp)
        
    sharp_k1, sharp_k2, sharp_k3 = sharp_levels_kernel(img)
    
    cv.imwrite(os.path.join(sharpened, "kernel_level_1.jpg"), sharp_k1)
    cv.imwrite(os.path.join(sharpened, "kernel_level_2.jpg"), sharp_k2)
    cv.imwrite(os.path.join(sharpened, "kernel_level_3.jpg"), sharp_k3)

if __name__ == '__main__':
    main()