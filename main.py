# Usage
# python main.py --image "image file"

# import the necessary packages
import cv2 as cv
import imutils
import argparse
import sys

# construct an argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="Path to input image")
args = vars(ap.parse_args())

# read the image from disk
image = cv.imread(args["image"])

# check if image was able read
if image is None:
    sys.exit("[INFO] image not found")

# resize the image, convert the image to grayscale, blur it
image = imutils.resize(image, width=400)
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
blur = cv.GaussianBlur(gray, (7, 7), 0)

# since our main focus is on the coins
# we use image segmentation (threshold) and use a bitwise and to retrieve only
# the coins i the image
thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)[1]
masked = cv.bitwise_and(image, image, mask=thresh)
edged = cv.Canny(masked, 30, 200)

# find the contours in the edge map and find the target one
# we assume is the outline of each coin
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# draw contours on each edge found and displays th number of coins found in the image
for c in cnts:
    area = cv.contourArea(c)
    print(area)
    cv.drawContours(image, [c], -1, (0, 255, 0), 2)
cv.putText(image, f"Coins: {len(cnts)}", (10, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# show the image and press any key to exit
cv.imshow("Output", image)
cv.waitKey(0)




