import numpy as np
from PIL import Image
from math import sqrt
import os
import cv2

RESULT_FOLDER = 'app/static/results'

def to_grayscale(image):
    return image.convert("L")

def apply_edge_detection(img_array, Sx, Sy):
    h, w = img_array.shape
    kh, kw = Sx.shape
    ph, pw = kh // 2, kw // 2
    result = np.zeros((h, w))

    for i in range(ph, h - ph):
        for j in range(pw, w - pw):
            region = img_array[i - ph:i + ph + 1, j - pw:j + pw + 1]
            if region.shape != Sx.shape:
                continue

            sum_x = np.sum(region * Sx)
            sum_y = np.sum(region * Sy)
            result[i, j] = min(255, sqrt(sum_x**2 + sum_y**2))

    result = 255 * (result / np.max(result)) if np.max(result) != 0 else result
    return Image.fromarray(result.astype(np.uint8))

def sobel_operator(image):
    img_array = np.array(to_grayscale(image))
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return apply_edge_detection(img_array, Sx, Sy)

def prewitt_operator(image):
    img_array = np.array(to_grayscale(image))
    Sx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Sy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return apply_edge_detection(img_array, Sx, Sy)

def roberts_operator(image):
    gray = np.array(to_grayscale(image), dtype=np.float32)
    Kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    Ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    gx = gray[:-1, :-1] * Kx[0, 0] + gray[:-1, 1:] * Kx[0, 1] + \
         gray[1:, :-1] * Kx[1, 0] + gray[1:, 1:] * Kx[1, 1]
    gy = gray[:-1, :-1] * Ky[0, 0] + gray[:-1, 1:] * Ky[0, 1] + \
         gray[1:, :-1] * Ky[1, 0] + gray[1:, 1:] * Ky[1, 1]
    
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = 255 * (magnitude / np.max(magnitude)) if np.max(magnitude) != 0 else magnitude
    result_img = Image.fromarray(magnitude.astype(np.uint8))
    return result_img


def canny_operator(image_path):
    base_name = os.path.basename(image_path) 
    result_filename = f"canny_{base_name}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite(result_path, edges)

def apply_selected_operator(image_path, operator_name, result_filename):
    image = Image.open(image_path)

    if operator_name == "sobel":
        result_img = sobel_operator(image)
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result_img.save(result_path)
    elif operator_name == "prewitt":
        result_img = prewitt_operator(image)
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result_img.save(result_path)
    elif operator_name == "roberts":
        result_img = roberts_operator(image)
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result_img.save(result_path)
    elif operator_name == "canny":
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        canny_operator(image_path)
    else:
        raise ValueError("Invalid operator selected")

    return result_filename
    
def apply_threshold(img_path, threshold, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, thresh_img)
