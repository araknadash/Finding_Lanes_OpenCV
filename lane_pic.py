import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('test_image.jpg')

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1) #fitting the X and Y co-ordinates with a 1 degree polynomial
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    #converting it into grayscale
    #lane_image = np.copy(image) #copying the test image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #smoothening the image and reduce noise
    #kernel of 5X5 is run on our gray image to smoothen out the edges with 0 deviation
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #Canny edge detection to detect edges which is done by running a derivative along
    #both x and y axis and lookout for strong derivatives to show major gradient change
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 10)
    return line_image
#Region of Interest to locate where the lanes are, we draw a traingle to focus on ROI
def roi(image):
    height = image.shape[0]
    polygons = np.array([[(200,height), (1100,height), (550,250)]])
    mask = np.zeros_like(image) #same amount as image
    cv2.fillPoly(mask, polygons, 255) #white triangle
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image
lane_image = np.copy(image)
canny = canny(image)
cropped_image = roi(canny)
# to detect straight lines in our ROI
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
averaged_lines = average_slope_intercept(lane_image, lines)
lines_image = display_lines(lane_image, averaged_lines)
combine_image = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)#blending the original image and line
cv2.imshow('result', combine_image)
cv2.waitKey(0) #used to display the image indefintely until we press any key
#edges detection can be done by detecting sharp change in brightness
