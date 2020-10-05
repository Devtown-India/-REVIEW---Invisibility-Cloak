"""
Created on Sat Sep 12 16:23:00 2020

@author: Ajay Bhat
"""

# Importing the necessary libraries
# 1.Computer Vision Library 
import cv2
# 2.numpy library
import numpy as np
# 3.time library
import time

#Step 1 - Capture and store the background frame

# The key idea is to replace the current frame pixels corresponding to the cloth with the background pixels to generate the effect of an invisibility cloak
# For this we need to store the the frame of the background

# Creating a VideoCapture object
# This will be used for image acquisition later in the code.
# If you want to try it on a video, just replace the 0 with a .mp4 file of your choice  
cap = cv2.VideoCapture(0)

# We give some time for the camera to warm up
time.sleep(3)

background=0

for i in range(30):
	ret,background = cap.read()
# In the above code, cap.read() method enables us to capture the latest frame(to be stored in a variable named 'background') with the camera and it also returns a boolean value(True/False) stored in the variable named 'ret'. Whenever a frame is reac correctly, 'ret' will be true.   

# But why capture background image using a ‘for loop’ ?
# Answer: As the background is static can’t we simply use a single frame? Sure, but the image captured is a bit dark compared to a multiple frames image. This is because the camera is just getting started on capturing frames and hence its parameters are not stable yet. Hence capturing multiple images of static background with a for loop does the trick.
#         Averaging over multiple frame also reduces the noise.


# Later invert the image/flip the image
background = np.flip(background,axis=1)

#--------------------------------------------------------------------------------------------------------------------------------

# Step 2 - Red Colour Detection

# Since we are focusing on a red cloth as invisibilty cloak, we will concentrate on detecting red colour in the frame.

# Note: Simply thresholding the R(red) channel will NOT work well since the RGB values are highly sensitive to illumination. As a result (inspite of the cloak being red in colour) there is a possibility, due to shadow in certain areas the red channel values being low.

# The correct approcah is to transform the color space from RGB to HSV

# HSV Color Space:
	# Hue : This channel encodes color information. Hue can be thought of an angle where 0 degree corresponds to the red color, 120 degrees corresponds to the green color, and 240 degrees corresponds to the blue color.
	# Saturation : This channel encodes the intensity/purity of color. For example, pink is less saturated than red.
	# Value : This channel encodes the brightness of color. Shading and gloss components of an image appear in this channel.

# The major advantage of using the HSV color space is that the color/tint/wavelength is represented by just the Hue component

# In the below code we first capture a live frame, convert the image from RGB to HSV color space and then define a specific range of H-S-V values to detect red color.
while(cap.isOpened()):
	#Capturing live frame
	ret, img = cap.read()
	
	# Laterally invert the image/flip the image
	img = np.flip(img,axis=1)
	
	# Converting image from BGR to HSV color space.
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	value = (35, 35)
	
	blurred = cv2.GaussianBlur(hsv, value,0)
	
	# Defining the lower range for red color detection.
	lower_red = np.array([0,120,70])
	upper_red = np.array([10,255,255])
	mask1 = cv2.inRange(hsv,lower_red,upper_red)
	
	# Defining upper range for red color detection
	lower_red = np.array([170,120,70])
	upper_red = np.array([180,255,255])
	mask2 = cv2.inRange(hsv,lower_red,upper_red)

	# The inRange function simply returns a binary mask, where white pixels (255) represent pixels that fall into the upper and lower limit range and black pixels (0) do not.	
	# The Hue values are actually distributed over a circle (range between 0-360 degrees) but in OpenCV to fit into 8bit value the range is from 0-180. The red color is represented by 0-30 as well as 150-180 values.
	# We use the range 0-10 and 170-180 to avoid detection of skin as red. High range of 120-255 for saturation is used because our cloth should be of highly saturated red color. The lower range of value is 70 so that we can detect red color in the wrinkles of the cloth as well.

	# Generating the final mask for red colour detection by addition of the two masks
	mask = mask1+mask2
	# Using the above line, we combine masks generated for both the red color range. It is basically doing an OR operation pixel-wise. 

	#---------------------------------------------------------

	# Steps 3 and 4- Segmenting out the detected red coloured cloth and displaying the output	
	# In the above step, we generated a mask to determine the region in the frame corresponding to the detected color. We refine this mask and then use it for segmenting out the cloth from the frame.
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
	
	# Replacing pixels corresponding to cloak(red cloth) with the background pixels.
	img[np.where(mask==255)] = background[np.where(mask==255)]
	cv2.imshow('Display',img)
	k = cv2.waitKey(10)
	if k == ord('q'):
		break
		
# Releasing the video capture object	
cap.release()
# Destroying all the generated windows
cv2.destroyAllWindows()