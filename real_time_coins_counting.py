# Usage
# python real_time_coins_counting.py --video "video file"

# import the necessary packages
import cv2 as cv
import imutils
import argparse

# construct an argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, type=str, help="path to input video stream")
args = vars(ap.parse_args())

# grab a pointer to the input video stream
print("[INFO] starting video stream")
vs = cv.VideoCapture(args["video"])

# loop over the frames of the video
while True:
    # read the frame from the video stream
    (grabbed, frame) = vs.read()

    # if the video frame was not grabbed then we have reached the end of the file
    if not grabbed:
        print("[INFO] no frame read from video stream exiting")
        break

    # resize the image, convert the image to grayscale, blur it
    image = imutils.resize(frame, width=300, height=300)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (7, 7), 0)

    # since our main focus is on the coins
    # we use image segmentation (threshold) and use a bitwise and to retrieve only
    # the coins i the image
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)

    # find the contours in the edge map and find the target one
    # we assume is the outline of each coin
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # initialise a counter for counting the number of coins
    count = 0

    # since we know most coins are circular in nature
    # we find each perimeter of the contour detected and check if is a circle
    # if the number of points detected in a contour is more than 5 then is a circle
    for c in cnts:
        epsilon = 0.01 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        if len(approx) > 5:

            # we draw a green circle around the detected coin
            # and we increase our counter by one
            cv.drawContours(image, [c], -1, (0, 255, 0), 2)
            count += 1

    cv.putText(image, f"Coins: {count}", (10, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output to our screen
    cv.imshow("Output", image)
    key = cv.waitKey(1) & 0xFF

    # if the q key is pressed break from the loop
    if key == ord("q"):
        break
cv.destroyAllWindows()
