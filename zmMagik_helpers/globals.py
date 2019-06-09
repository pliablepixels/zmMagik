import cv2
import numpy as np


set_frames = {}
mon_list = []
args = []
template = None
logger = None  # loggin handler
remove_downloaded = False # true, if the input was a remote url
out_file = None # video file frames will be written to
json_out = []
orig_fps = None


poly_mask = None
raw_poly_mask = None
