import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log
import cv2
import numpy as np
from shapely.geometry import Polygon
import dateparser 
from datetime import datetime, timedelta

class DetectBackground:
    def __init__(self, min_accuracy=0.7, min_blend_area=2500, kernel_fill=40, dist_threshold=15000, history=120):
        self.min_accuracy = max (min_accuracy, 0.7)
        self.min_blend_area = min_blend_area
        self.kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.kernel_fill = np.ones((kernel_fill,kernel_fill),np.uint8)
        self.dist_threshold = dist_threshold
        self.history = history

        '''
        c = """
        DetectBackground init:
        minimum accuracy={}
        minimum blend={}
        fill for foreground={}
        distance={}
        history={}
        """.format(min_accuracy,min_blend_area,kernel_fill, dist_threshold, history)
        utils.dim_print(c)
        '''

        # read https://docs.opencv.org/3.3.0/d2/d55/group__bgsegm.html#gae561c9701970d0e6b35ec12bae149814

        #self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=30, nmixtures=5, backgroundRatio=0.8, noiseSigma=0) 
        #self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(decisionThreshold=0.98, initializationFrames=10)
        #self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=20) 
        #self.fgbg=cv2.bgsegm.createBackgroundSubtractorGSOC(noiseRemovalThresholdFacBG=0.01, noiseRemovalThresholdFacFG=0.0001)
        #self.fgbg=cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 5, useHistory = True, maxPixelStability = 5 *60,isParallel = True)
        self.fgbg=cv2.createBackgroundSubtractorKNN(detectShadows=False, history=self.history, dist2Threshold = self.dist_threshold)
        #fgbg=cv2.bgsegm.createBackgroundSubtractorLSBP()

    def detect(self,frame, frame_b, frame_cnt, orig_fps, starttime):
        frame_mask = self.fgbg.apply(frame)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, self.kernel_clean)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, self.kernel_fill)
        frame_mask = cv2.medianBlur(frame_mask,15)

        # foreground extraction of new frame
        foreground_a = cv2.bitwise_and(frame,frame, mask=frame_mask)

        # clear out parts on blended frames where foreground will be added
        frame_mask_inv = cv2.bitwise_not(frame_mask)
        # blend frame with foreground a missing
        modified_frame_b = cv2.bitwise_and(frame_b, frame_b, mask=frame_mask_inv)
        # new frame with foreground missing
        #modified_frame = cv2.bitwise_and(frame, frame, mask=frame_mask_inv)
        #modified_frame_b = cv2.addWeighted(modified_frame_b, 0.5, modified_frame, 0.5, 0)

        ctrs,_ =  cv2.findContours(frame_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        merged_frame = cv2.add(modified_frame_b, foreground_a)
        relevant = False;
        for c in ctrs:
            area = cv2.contourArea(c)
            if area >= self.min_blend_area:
                relevant = True
                x,y,w,h = cv2.boundingRect(c)
                pts = Polygon([[x,y], [x+w,y], [x+w, y+h], [x,y+h]])
                #print (pts)
                if g.poly_mask is None or g.poly_mask.intersects(pts):
                    text = '{}s, Frame: {}'.format(int(frame_cnt/orig_fps), frame_cnt)
                    if starttime:
                        st = dateparser.parse(starttime)
                        #from_time = to_time - datetime.timedelta(hours = 1)
                        # print (st)
                        dt = st + timedelta(seconds=int(frame_cnt/orig_fps))
                        text = dt.strftime('%b %d, %I:%M%p')
                    
                    text = text.upper()
                    (tw, th) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=g.args['fontscale'], thickness=2)[0]

                    loc_x1 = x
                    loc_y1 = y - th - 4
                    loc_x2 = x + tw + 4
                    loc_y2 = y


                    cv2.rectangle(merged_frame, (loc_x1, loc_y1), (loc_x1+tw+4,loc_y1+th+4), (0,0,0), cv2.FILLED)
                    cv2.putText(merged_frame, text, (loc_x1+2, loc_y2-2), cv2.FONT_HERSHEY_PLAIN, fontScale=g.args['fontscale'], color=(255,255,255), thickness=1)
                    #cv2.rectangle(merged_frame,(x,y),(x+w,y+h),(0,255,0),2)
        #      final frame,  extracted fg, mask
        return merged_frame, foreground_a, frame_mask, relevant
