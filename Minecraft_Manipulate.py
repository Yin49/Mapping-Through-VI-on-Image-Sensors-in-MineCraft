import time
import pynput
from pynput.keyboard import Key, Controller
mouse = pynput.mouse.Controller()

time.sleep(6)


keyboard = Controller()

# Press the "A" key
keyboard.press('a')

time.sleep(2)

# Release the "A" key
keyboard.release('a')

def move_smooth(xm, ym, t):
    for i in range(t):
        if i < t/2:
            h = i
        else:
            h = t - i
        mouse.move(h*xm, 0)
        time.sleep(1/60)

# turn 180 degrees
move_smooth(2, 2, 40)



import numpy as np
import cv2
import pyautogui
import matplotlib.pyplot as plt
   
  
# take screenshot using pyautogui
image = pyautogui.screenshot()
   
# since the pyautogui takes as a 
# PIL(pillow) and in RGB we need to 
# convert it to numpy array and BGR 
# so we can write it to the disk
image = cv2.cvtColor(np.array(image),
                     cv2.COLOR_RGB2BGR)


# Get the original height and width of the image
height, width = image.shape[:2]

# Define the new dimensions
new_height = int(height / 2)
new_width = int(width / 2)

# Resize the image using cv2.resize()
resized_img = cv2.resize(image, (new_width, new_height))


# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Show the original image in the first subplot
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')

# Show the resized image in the second subplot
axs[1].imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
axs[1].set_title('Resized')

# Display the figure
plt.show()



   
# writing it to the disk using opencv
# cv2.imwrite("image1.png", image)











# from mcpi.minecraft import Minecraft
# from minecraftstuff import MinecraftShape

# serverAddress="192.168.1.94" # change to your minecraft server
# pythonApiPort=4711 #default port for RaspberryJuice plugin is 4711, it could be changed in plugins\RaspberryJuice\config.yml
# playerName="David_Love_Robot" # change to your username

# mc = Minecraft.create()
# pos = mc.player.getPos()

# print("pos: x:{},y:{},z:{}".format(pos.x,pos.y,pos.z))


# myShape = MinecraftShape(mc, pos)


# # create a sign block with the command
# mc.setBlock(pos.x, pos.y, pos.z, 63)
# mc.setSign(pos.x, pos.y, pos.z, 63, 0, "/tp @s 411 70 654 facing 413 70 654")

# execute the command on the sign
# mc.setCommand(pos.x, pos.y, pos.z, "setblock ~ ~ ~ minecraft:air")


# direction = mc.player.getRotation()
# pitch = mc.player.getPitch()


# print("direction:{}".format(direction))
# print("pitch:{}".format(pitch))
# x,y,z = pos = mc.player.getTilePos()
# mc.player.setTilePos(0, 100, 0)
# mc.player.setTilePos(x+2,y,z)

