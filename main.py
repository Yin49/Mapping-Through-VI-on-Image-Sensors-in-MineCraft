import numpy as np
import cv2
from segmentation import segmentation_task
from PIL import Image

# import Minecraft_Manipulate as Minecraft



def read_image():
    train_set = []

    for i in range(12, 22):
        filename = f"{i}.png"
        path = f'C:/Users/12866/Desktop/minecraft_project/screenshots/{filename}'
        image = Image.open(path)
        train_set.append(image)

    
    return train_set


def main():

    train_set = read_image()
    segmentation_task(train_set, 15)

    # teleport to integer position
    # Minecraft.teleport_to_integer_coords()


    
    

main()
