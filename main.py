# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 10:42:33 2023

@author: Ivan
"""
import tkinter as tk
from tkinter import *
from lane_detection.lane import *
import imutils


vidcap = cv2.VideoCapture("archive/video-sd.mp4")

if (vidcap.isOpened()== False): 
  print("Error opening video stream or file")

success, image = vidcap.read()
if success == True:
    img_bgr = imutils.resize(image, width=800)
     
  
#file='road2.jpg'
#img_bgr = cv2.imread(file)
#overlay= img_bgr.copy()
class RoiPoints:
    # 'what' and 'why' should probably be fetched in a different way, suitable to the app
    def __init__(self, heightUp, heightDown, my_scale1, my_scale2, my_scale3, my_scale4):
        self.heightUp = heightUp
        self.heightDown = heightDown
        self.my_scale1= my_scale1
        self.my_scale2= my_scale2
        self.my_scale3= my_scale3
        self.my_scale4= my_scale4
        

my_w = tk.Tk()
my_w.geometry("400x400") 

my_w.title("Set points")

tk.Label(my_w, text="Set up-height:").grid(row=1, column=0)
tk.Label(my_w, text="Set down-height :").grid(row=3, column=0)
tk.Label(my_w, text="Gore-lijevo:").grid(row=5, column=0)
tk.Label(my_w, text="Gore-desno:").grid(row=7, column=0)
tk.Label(my_w, text="Dolje-desno:").grid(row=9, column=0)
tk.Label(my_w, text="Dolje-lijevo:").grid(row=11, column=0)

def onChange(value=None):
    global img_persp, img_warped
    compute_ROI(img_bgr,heightUp.get(), heightDown.get(), my_scale1.get(),my_scale2.get(),my_scale3.get(),my_scale4.get() )
    img_persp, img_warped = warp(img_bgr)
    cv2.imshow("Region of interest", img_persp)
    
def exit(value=None):
    my_w.destroy()
    cv2.destroyAllWindows()

    
heightUp = tk.Scale(my_w, from_=0, to=100, orient='horizontal',command=onChange)
heightUp.grid(row=1,column=1) 
heightUp.set(56)


heightDown = tk.Scale(my_w, from_=0, to=100, orient='horizontal',command=onChange)
heightDown.grid(row=3,column=1) 
heightDown.set(68)

my_scale1 = tk.Scale(my_w, from_=0, to=100, orient='horizontal',command=onChange)
my_scale1.grid(row=5,column=1) 
my_scale1.set(44)

my_scale2 = tk.Scale(my_w, from_=0, to=100, orient='horizontal',command=onChange)
my_scale2.grid(row=7,column=1) 
my_scale2.set(56)


my_scale3 = tk.Scale(my_w, from_=0, to=100, orient='horizontal',command=onChange)
my_scale3.grid(row=9,column=1) 
my_scale3.set(81)


my_scale4 = tk.Scale(my_w, from_=0, to=100, orient='horizontal',command=onChange)
my_scale4.grid(row=11,column=1) 
my_scale4.set(25)


exit_button = tk.Button(my_w, text="Save", command=exit)
exit_button.grid(row=16,column=1) 

points1=RoiPoints(heightUp.get(), heightDown.get(), my_scale1.get(),my_scale2.get(),my_scale3.get(),my_scale4.get())
my_w.mainloop()

########

img_hls = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS)
img_edge = edge_detection(img_warped[:, :, 1])


def onChangeThreshold(value=None):
    global img_binary_combined, img_binary_solo
    (img_binary_combined, img_binary_solo) = threshold(img_hls[:, :, 1], img_edge,threshold_up.get(),threshold_down.get(),threshold_break.get())
    cv2.imshow("Threshold", img_binary_combined)
    cv2.imshow("Warped", cv2.cvtColor(img_warped, cv2.COLOR_RGB2BGR))


class ThresholdPoints:
    # 'what' and 'why' should probably be fetched in a different way, suitable to the app
    def __init__(self, threshold_up, threshold_down, threshold_break):
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down
        self.threshold_break= threshold_break


my_w = tk.Tk()
my_w.geometry("400x400") 

my_w.title("Set threshold")

tk.Label(my_w, text="Threshold down:").grid(row=1, column=0)
tk.Label(my_w, text="Threshold up:").grid(row=3, column=0)
tk.Label(my_w, text="threshold break:").grid(row=5, column=0)


threshold_up = tk.Scale(my_w, from_=0, to=255, orient='horizontal',command=onChangeThreshold)
threshold_up.grid(row=5,column=1) 
threshold_up.set(15)

threshold_down = tk.Scale(my_w, from_=0, to=255, orient='horizontal',command=onChangeThreshold)
threshold_down.grid(row=7,column=1) 
threshold_down.set(60)


threshold_break = tk.Scale(my_w, from_=0, to=255, orient='horizontal',command=onChangeThreshold)
threshold_break.grid(row=9,column=1) 
threshold_break.set(140)


exit_button = tk.Button(my_w, text="Save", command=exit)
exit_button.grid(row=16,column=1) 

points2= ThresholdPoints(threshold_up.get(), threshold_down.get(), threshold_break.get())
my_w.mainloop()

hist = histogram(img_binary_combined)
lanes = lanes_full_histogram(hist)

ret, sw = slide_window(img_warped, img_binary_combined, lanes, 15)

left_lanes = deepcopy(sw.left)
right_lanes = deepcopy(sw.right)

img_lane, img_lane_orig = show_lanes(sw, img_warped, img_bgr)


alpha = 0.70


# Check if camera opened successfully
if (vidcap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(vidcap.isOpened()):

  # Capture frame-by-frame
  ret, frame = vidcap.read()
  if ret == True:
    frame = imutils.resize(frame, width=800)
    overlay= frame.copy()

    
    compute_ROI(frame,getattr(points1, "heightUp"), getattr(points1, "heightDown"), getattr(points1, "my_scale1"), getattr(points1, "my_scale2"),getattr(points1, "my_scale3"),getattr(points1, "my_scale4") )
    img_persp, img_warped = warp(frame)
    img_hls = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS)
    img_edge = edge_detection(img_warped[:, :, 1])
    (img_binary_combined, img_binary_solo) = threshold(img_hls[:, :, 1], img_edge,getattr(points2,"threshold_up"),getattr(points2,"threshold_down"),getattr(points2,"threshold_break"))

    hist = histogram(img_binary_combined)
    lanes = lanes_full_histogram(hist)
    ret, sw = slide_window(img_warped, img_binary_combined, lanes, 15)
    
    img_lane, img_lane_orig = show_lanes(sw, img_warped, frame)

    result = cv2.addWeighted(overlay, alpha, img_lane_orig, 1 - alpha, 0)

    # Display the resulting frame
    cv2.imshow('Frame',result)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 

