import cv2
import numpy as np

MIN_ACCURACY = 0.7

set_frames = {}
mon_list = []
args = []
template = None
logger = None  # loggin handler
remove_downloaded = False # true, if the input was a remote url
out_file = None # video file frames will be written to
json_out = []
kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_fill = np.ones((20,20),np.uint8)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False) 
poly_mask = None
