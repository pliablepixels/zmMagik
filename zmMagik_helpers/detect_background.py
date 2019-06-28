import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log
import cv2
import numpy as np
from shapely.geometry import Polygon
import dateparser 
from datetime import datetime, timedelta

class DetectBackground:
    # unspecified defaults come from config defaults
    def __init__(self, min_accuracy, min_blend_area, kernel_fill=20, dist_threshold=15000, history=400):
        self.min_accuracy = max (min_accuracy, 0.7)
        self.min_blend_area = min_blend_area
        self.kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
        self.kernel_fill = np.ones((kernel_fill,kernel_fill),np.uint8)
        self.dist_threshold = dist_threshold
        self.history = history
   


        # read https://docs.opencv.org/3.3.0/d2/d55/group__bgsegm.html#gae561c9701970d0e6b35ec12bae149814

        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=self.history, nmixtures=5, backgroundRatio=0.7, noiseSigma=0) 
        #self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(decisionThreshold=0.98, initializationFrames=10)
        #self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=self.history) 
        #self.fgbg=cv2.bgsegm.createBackgroundSubtractorGSOC(noiseRemovalThresholdFacBG=0.01, noiseRemovalThresholdFacFG=0.0001)
        #self.fgbg=cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 5, useHistory = True, maxPixelStability = 5 *60,isParallel = True)
        #self.fgbg=cv2.createBackgroundSubtractorKNN(detectShadows=False, history=self.history, dist2Threshold = self.dist_threshold)
        #fgbg=cv2.bgsegm.createBackgroundSubtractorLSBP()

        utils.success_print('Background subtraction initialized')

    def detect(self,frame, frame_b, frame_cnt, orig_fps, starttime, set_frames):
       
        #frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #frame_hsv[:,:,0] = 0 # see if removing intensity helps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # create initial background subtraction
        frame_mask = self.fgbg.apply(gray)
        # remove noise
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, self.kernel_clean)
        # blur to merge nearby masks, hopefully
        frame_mask = cv2.medianBlur(frame_mask,15)
        #frame_mask = cv2.GaussianBlur(frame_mask,(5,5),cv2.BORDER_DEFAULT)
        #frame_mask = cv2.blur(frame_mask,(20,20))
        
        h,w,_ = frame.shape
        new_frame_mask = np.zeros((h,w),dtype=np.uint8)
        copy_frame_mask = frame_mask.copy()
        # find contours of mask
        relevant = False
        ctrs,_ =  cv2.findContours(copy_frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        # only select relevant contours 
        for contour in ctrs:
            area = cv2.contourArea(contour)
            if area >= self.min_blend_area:
                x,y,w,h = cv2.boundingRect(contour)
                pts = Polygon([[x,y], [x+w,y], [x+w, y+h], [x,y+h]])
                if g.poly_mask is None or g.poly_mask.intersects(pts):
                    relevant = True
                    cv2.drawContours(new_frame_mask, [contour], -1, (255, 255, 255), -1)
                    rects.append([x,y,w,h])
                    
        # do a dilation to again, combine the contours
        frame_mask = cv2.dilate(new_frame_mask,self.kernel_fill,iterations = 5)
        frame_mask = new_frame_mask

      
        # foreground extraction of new frame
        foreground_a = cv2.bitwise_and(frame,frame, mask=frame_mask)
        # clear out parts on blended frames where foreground will be added
        frame_mask_inv = cv2.bitwise_not(frame_mask)
        # blend frame with foreground a missing
    
        
        modified_frame_b = cv2.bitwise_and(frame_b, frame_b, mask=frame_mask_inv)
        merged_frame = cv2.add(modified_frame_b, foreground_a)


         # draw mask on blend frame
        cv2.polylines(merged_frame, [g.raw_poly_mask], True, (0,0,255), thickness=1)

        # now draw times
        boxed_frame = frame.copy()
        for rect in rects:
            x,y,w,h = rect
            obj_info = {
                'name': 'object',
                'time':int(frame_cnt/orig_fps),
                'frame': frame_cnt,
                'location': ((x,y),(x+w, y+h)),
                'confidence': None
            }
            # draw blue boxes after all intelligence is done
            cv2.rectangle(boxed_frame, (x, y), (x+w, y+h), (255,0,0), 2)
            text = '{}s, Frame: {}'.format(int(frame_cnt/orig_fps), frame_cnt)
            if starttime:
                st = dateparser.parse(starttime)
                #from_time = to_time - datetime.timedelta(hours = 1)
                # print (st)
                dt = st + timedelta(seconds=int(frame_cnt/orig_fps))
                text = dt.strftime('%b %d, %I:%M%p')
                obj_info['time'] = text
            text = text.upper()
            utils.write_text(merged_frame, text, x,y)
            if g.args['detection_type'] != 'mixed':
                # if its mixed, Yolo will write this
                set_frames['frames'].append (obj_info)
                if g.args['drawboxes']:
                    cv2.rectangle(merged_frame, (x, y), (x+w, y+h), (255,0,0), 2)
        
        return merged_frame, foreground_a, frame_mask, relevant, boxed_frame
