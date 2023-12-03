import numpy as np
from PIL import Image

def load_image_to_gray(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)
    return image

def resize_image(image, size):
    resized_image = np.resize(image, size)
    return resized_image

image_path = '1.jpg'

gray_image = load_image_to_gray(image_path) 
  
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.imshow(gray_image, cmap='gray')
plt.show()
