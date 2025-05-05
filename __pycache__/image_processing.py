import numpy as np
from PIL import Image
from math import sqrt

def to_grayscale(image):
    return image.convert("L")

def apply_edge_detection(img_array, Sx, Sy):
    h, w = img_array.shape
    sx, sy = Sx.shape[0] // 2, Sy.shape[0] // 2
    result = np.zeros((h, w))

    for i in range(sx, h - sx):
        for j in range(sy, w - sy):
            sum_x = np.sum(img_array[i-sx:i+sx+1, j-sy:j+sy+1] * Sx)
            sum_y = np.sum(img_array[i-sx:i+sx+1, j-sy:j+sy+1] * Sy)
            result[i, j] = min(255, sqrt(sum_x**2 + sum_y**2))
    
    return Image.fromarray(result.astype(np.uint8))

def sobel_operator(image):
    grayscale_image = to_grayscale(image)  # Convert to grayscale
    img_array = np.array(grayscale_image)  # Convert to numpy array
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return apply_edge_detection(img_array, Sx, Sy)

def prewitt_operator(image):
    grayscale_image = to_grayscale(image)  # Convert to grayscale
    img_array = np.array(grayscale_image)  # Convert to numpy array
    Sx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Sy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return apply_edge_detection(img_array, Sx, Sy)

def roberts_operator(image):
    grayscale_image = to_grayscale(image)  # Convert to grayscale
    img_array = np.array(grayscale_image)  # Convert to numpy array
    Sx = np.array([[1, 0], [0, -1]])
    Sy = np.array([[0, 1], [-1, 0]])
    return apply_edge_detection(img_array, Sx, Sy)