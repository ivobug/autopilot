# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:22:36 2023

@author: Ivan
"""

import cv2
import numpy as np
from copy import deepcopy

from matplotlib.pyplot import imshow, figure, plot


perspective_correction = None
perspective_correction_inv = None
perspective_trapezoid = None
warp_size = None
orig_size = None
left_fit_avg = None
right_fit_avg = None
left_fit=None
right_fit=None
MIN_DETECTIONS = 8
MAX_DETECTIONS = 10
font = cv2.FONT_HERSHEY_SIMPLEX
wheel=cv2.imread('src/steering-wheel.jpg')
# Defining variables to hold meter-to-pixel conversion
ym_per_pix = 30 / 720
# Standard lane width is 3.7 meters divided by lane width in pixels which is
# calculated to be approximately 720 pixels not to be confused with frame height
xm_per_pix = 3.7 / 720

# pt1, pt2, ptr3, and pt4 and four points defining a trapezoid used for the perspective correction
def compute_ROI(image, height1, height2 , lup, rup, rdown, ldown ):
    global perspective_trapezoid, perspective_dest

    height = image.shape[0]
    width = image.shape[1]
    
    height2=round(height2*height/100)
    height1=round(height1*height/100)
    lup=round(lup*width/100)
    rup=round(rup*width/100)
    rdown=round(rdown*width/100)
    ldown=round(ldown*width/100)
    
    pt1=[ldown, height2]
    pt2=[lup, height1]
    pt3= [rup, height1]
    pt4=[rdown,height2]
    
    
    
    perspective_trapezoid = [(pt1[0], pt1[1]), (pt2[0], pt2[1]), (pt3[0], pt3[1]), (pt4[0], pt4[1])]
    src = np.float32([pt1, pt2, pt3, pt4])
    # widest side on the trapezoid
    x1 = pt1[0]
    x2 = pt4[0]
    # height of the trapezoid
    y1 = pt1[1]
    y2 = pt2[1]
    h = y1 - y2
    # The destination is a rectangle with the height of the trapezoid and the width of the widest side
    dst = np.float32([[x1, h], [x1, 0], [x2, 0], [x2, h]])
    perspective_dest = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

    global perspective_correction, perspective_correction_inv
    global warp_size, orig_size

    perspective_correction = cv2.getPerspectiveTransform(src, dst)
    perspective_correction_inv = cv2.getPerspectiveTransform(dst, src)
    warp_size = (width, h)
    orig_size = (width, height)

def warp(img):
    img_persp = img.copy()
    cv2.line(img_persp, perspective_dest[0], perspective_dest[1], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[1], perspective_dest[2], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[2], perspective_dest[3], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[3], perspective_dest[0], (255, 255, 255), 3)

    cv2.line(img_persp, perspective_trapezoid[0], perspective_trapezoid[1], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[1], perspective_trapezoid[2], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[2], perspective_trapezoid[3], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[3], perspective_trapezoid[0], (0, 192, 0), 3)

    return img_persp, cv2.warpPerspective(img, perspective_correction, warp_size, flags=cv2.INTER_LANCZOS4)





def edge_detection(channel):
    edge_x = cv2.Scharr(channel, cv2.CV_64F, 1, 0)  # Edge detection using the Scharr operator
    edge_x = np.absolute(edge_x)

    return np.uint8(255 * edge_x / np.max(edge_x))

#img_hls = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS).astype(np.float)
#img_edge = edge_detection(img_warped[:, :, 1])
###############
#figure()
#imshow(cv2.cvtColor(img_edge, cv2.COLOR_RGB2BGR))



def threshold(channel_threshold, channel_edge,threshold_up,threshold_down,threshold_break ):
    # Gradient threshold
    binary = np.zeros_like(channel_edge)
    height = binary.shape[0]

    threshold_up = threshold_up
    threshold_down = threshold_down
    threshold_delta = threshold_down - threshold_up

    for y in range(height):
        binary_line = binary[y, :]
        edge_line = channel_edge[y, :]
        threshold_line = threshold_up + threshold_delta * y / height
        binary_line[edge_line >= threshold_line] = 255

    binary[(channel_threshold >= threshold_break) & (channel_threshold <= 255)] = 255

    binary_threshold = np.zeros_like(channel_threshold)
    binary_threshold[(channel_threshold >= threshold_break) & (channel_threshold <= 255)] = 255

    return binary, binary_threshold




def histogram(img):
    partial_img = img[img.shape[0] * 2 // 3:, :]  # Select the bottom part (one third of the image)
    hist = np.sum(partial_img, axis=0)
    ##########################
    figure()
    #plot(hist)

    return hist


class HistLanes:
    def __init__(self, x_left, x_right, left_confidence, right_confidence):
        self.x_left = x_left
        self.x_right = x_right
        self.left_confidence = left_confidence
        self.right_confidence = right_confidence

def lanes_full_histogram(histogram):
    size = len(histogram)
    max_index_left = np.argmax(histogram[0:size // 2])
    max_index_right = np.argmax(histogram[size // 2:]) + size // 2

    return HistLanes(max_index_left, max_index_right, histogram[max_index_left], histogram[max_index_right])

def moving_average(prev_average, new_value, beta):
    return beta * prev_average + (1 - beta) * new_value if prev_average is not None else new_value

# Single lane line
class Line:
    lane_indexes = None
    # pixel positions
    x = None
    y = None

    # Fit a second order polynomial to each
    fit = None
    # Plotting parameters
    fitx = None

    # Histogram
    hist_x = None


# Data collected during the sliding windows phase
class SlideWindow:
    left = Line()
    right = Line()
    hist = None

    left_avg = None
    right_avg = None

    ploty = None

    def __init__(self, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y):
        self.left.lane_indexes = np.concatenate(left_lane_indexes)
        self.right.lane_indexes = np.concatenate(right_lane_indexes)
        self.left.hist_x = hist.x_left
        self.right.hist_x = hist.x_right
        # Extract left and right positions
        self.left.x = non_zero_x[self.left.lane_indexes]
        self.left.y = non_zero_y[self.left.lane_indexes]
        self.right.x = non_zero_x[self.right.lane_indexes]
        self.right.y = non_zero_y[self.right.lane_indexes]

    def plot_lines(self, img, color_left=(0, 255, 255), color_right=(0, 255, 255)):
        left = []
        right = []
        for i in range(0, len(self.ploty)):
            left.append((self.left.fitx[i], self.ploty[i]))
            right.append((self.right.fitx[i], self.ploty[i]))

        cv2.polylines(img, np.int32([left]), False, color_left)
        cv2.polylines(img, np.int32([right]), False, color_right)

        return img


def fit_slide_window(binary_warped, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y):
    sw = SlideWindow(hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y)

    # y coordinates
    sw.ploty = np.array([float(x) for x in range(binary_warped.shape[0])])

    if len(sw.left.y) == 0:
        return False, sw
    

    # Fit a second order polynomial to approximate the points
    left_fit = np.polynomial.polynomial.polyfit(sw.left.y, sw.left.x, 2)
    right_fit = np.polynomial.polynomial.polyfit(sw.right.y, sw.right.x, 2)

    global left_fit_avg, right_fit_avg

    left_fit_avg = moving_average(left_fit_avg, left_fit, 0.92)
    right_fit_avg = moving_average(right_fit_avg, right_fit, 0.92)

    # Generate list of x and y values, using the terms of the polynomial
    # x = Ay^2 + By + C;
    sw.left.fitx = left_fit_avg[2] * sw.ploty ** 2 + left_fit_avg[1] * sw.ploty + left_fit_avg[0]
    sw.right.fitx = right_fit_avg[2] * sw.ploty ** 2 + right_fit_avg[1] * sw.ploty + right_fit_avg[0]
    
    ltx = np.trunc(sw.left.fitx)
    rtx = np.trunc(sw.right.fitx)

    return True, sw ,ltx, rtx, left_fit, right_fit

def slide_window(img, binary_warped, hist, num_windows):
    img_height = binary_warped.shape[0]
    window_height = np.int(img_height / num_windows)
    # Indices (e.g. coordinates) of the pixels that are not zero
    non_zero = binary_warped.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])
    # Current positions, to be updated while sliding the window; we start with the ones identified with the histogram
    left_x = hist.x_left
    right_x = hist.x_right
    margin = 80
    # Set minimum number of pixels found to recenter window
    min_pixels = 25
    left_lane_indexes = []
    right_lane_indexes = []

    out_img = img.copy()

    for idx_window in range(num_windows):
        # X range where we expect the left lane to land
        win_x_left_min = left_x - margin
        win_x_left_max = left_x + margin
        # X range where we expect the right lane to land
        win_x_right_min = right_x - margin
        win_x_right_max = right_x + margin
        # Y range that we are analyzing
        win_y_top = img_height - idx_window * window_height
        win_y_bottom = win_y_top - window_height

        # Show the windows
        cv2.rectangle(out_img, (win_x_left_min, win_y_bottom), (win_x_left_max, win_y_top), (255, 255, 255), 2)
        cv2.rectangle(out_img, (win_x_right_min, win_y_bottom), (win_x_right_max, win_y_top), (255, 255, 255), 2)

        # Non zero pixels in x and y
        non_zero_left = ((non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top) & (non_zero_x >= win_x_left_min) & (
                non_zero_x < win_x_left_max)).nonzero()[0]
        non_zero_right = ((non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top) & (non_zero_x >= win_x_right_min) & (
                non_zero_x < win_x_right_max)).nonzero()[0]

        left_lane_indexes.append(non_zero_left)
        right_lane_indexes.append(non_zero_right)
        # If you found > min_pixels pixels, recenter next window on the mean position
        if len(non_zero_left) > min_pixels:
            left_x = np.int(np.mean(non_zero_x[non_zero_left]))

        if len(non_zero_right) > min_pixels:
            right_x = np.int(np.mean(non_zero_x[non_zero_right]))

    valid, sw , leftx,rightx, left_fit, right_fit= fit_slide_window(binary_warped, hist, left_lane_indexes, right_lane_indexes, non_zero_x, non_zero_y)

    if valid :
        out_img[non_zero_y[sw.left.lane_indexes], non_zero_x[sw.left.lane_indexes]] = [0, 255, 192]
        out_img[non_zero_y[sw.right.lane_indexes], non_zero_x[sw.right.lane_indexes]] = [0, 255, 192]
        img_plot = sw.plot_lines(out_img)
        #####################################
        #figure()
        #imshow(img_plot)

    return valid, sw, leftx ,rightx, sw.ploty, left_fit, right_fit

#lanes = lanes_full_histogram(hist)

#ret, sw = slide_window(img_warped, img_binary_combined, lanes, 15)

def general_search(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
 
    #right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = leftx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = left_fitx
    ret['ploty'] = ploty

    return ret

def measure_lane_curvature(ploty, leftx, rightx):

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)


    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    # Decide if it is a left or a right curve
        
    if leftx[-1] - leftx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Left Curve'

    return (left_curverad ) / 2.0, curve_direction
    


def show_lanes(sw, img_warped, img_orig):
    img = img_warped.copy()

    if sw.left:
        fitx_points_warped = np.float32([np.transpose(np.vstack([sw.left.fitx, sw.ploty]))])
        fitx_points = cv2.perspectiveTransform(fitx_points_warped, perspective_correction_inv)
        left_line_warped = np.int_(fitx_points_warped[0])
        left_line = np.int_(fitx_points[0])
        n = len(left_line)

        for i in range(n - 1):
            cv2.line(img_orig, (left_line[i][0], left_line[i][1]), (left_line[i + 1][0], left_line[i + 1][1]),
                     (0, 255, 0), 5)
            cv2.line(img, (left_line_warped[i][0], left_line_warped[i][1]),
                     (left_line_warped[i + 1][0], left_line_warped[i + 1][1]), (0, 255, 0), 5)

    if sw.right:
        fitx_points_warped = np.float32([np.transpose(np.vstack([sw.right.fitx, sw.ploty]))])
        fitx_points = cv2.perspectiveTransform(fitx_points_warped, perspective_correction_inv)
        right_line_warped = np.int_(fitx_points_warped[0])
        right_line = np.int_(fitx_points[0])

        for i in range(len(right_line) - 1):
            cv2.line(img_orig, (right_line[i][0], right_line[i][1]), (right_line[i + 1][0], right_line[i + 1][1]),
                     (0, 0, 255), 5)
            cv2.line(img, (right_line_warped[i][0], right_line_warped[i][1]),
                     (right_line_warped[i + 1][0], right_line_warped[i + 1][1]), (0, 0, 255), 5)
    cv2.fillPoly(img_orig, [np.array([[right_line[0][0], right_line[0][1]], [right_line[-1][0], right_line[-1][1]], [left_line[-1][0], left_line[-1][1]],[left_line[0][0], left_line[0][1]]])], (0, 255, 0))


    return img, img_orig

def plotSteeringInfo(img_orig,radius, direction):
    radius=radius-25
    wheel_radius=radius
    if radius<0:
        wheel_radius=360+radius
    if (direction != 'Straight'):
        text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius)
        text1 = 'Curve Direction: ' + (direction)

    else:
        text = 'Radius of Curvature: ' + 'N/A'
        text1 = 'Curve Direction: ' + (direction)

    cv2.putText(img_orig, text , (10,20), font, 0.4, (255,255, 255), 0, cv2.LINE_AA)
    cv2.putText(img_orig, text1, (10,35), font, 0.4, (255,255, 255), 0, cv2.LINE_AA)

    overlay_img1 = np.ones(img_orig.shape,np.uint8)*255

    small_img=cv2.imread('src/steering-wheel.jpg')
    small_img = cv2.resize(small_img, (50,50), interpolation = cv2.INTER_AREA)
    print(radius)
    M = cv2.getRotationMatrix2D((50/2,50/2),-(wheel_radius),1) 
    small_img = cv2.warpAffine(small_img,M,(50,50)) 
    rows,cols,channels = small_img.shape
    overlay_img1[45:rows+45, 10:cols+10 ] = small_img

    img2gray = cv2.cvtColor(overlay_img1,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray,220,55,cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    temp1 = cv2.bitwise_and(img_orig,img_orig,mask = mask_inv)
    temp2 = cv2.bitwise_and(overlay_img1,overlay_img1, mask = mask)

    img_orig = cv2.add(temp1,temp2)


    return img_orig



