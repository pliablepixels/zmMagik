import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log
import cv2
import numpy as np
from shapely.geometry import Polygon
import dateparser
from datetime import datetime, timedelta
import zmMagik_helpers.simpleyolo.simpleYolo as yolo
from ctypes import *

# credit: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class DetectYolo:

    def __init__(self,configPath=None, weightPath=None, labelsPath=None, darknetLib=None, kernel_fill=3):

        if g.args['gpu']:
            utils.success_print('Using GPU model for YOLO')
            utils.success_print('If you run out of memory, please tweak yolo.cfg')
            self.m = yolo.SimpleYolo(configPath=configPath,
                    weightPath=weightPath,
                    darknetLib=darknetLib,
                    labelsPath=labelsPath,
                    useGPU=True)
        else:
            utils.success_print('Using CPU/OpenCV model for YOLO')
            self.net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
            self.labels = open(labelsPath).read().strip().split("\n")
            np.random.seed(42)
            self.colors = np.random.randint(
                0, 255, size=(len(self.labels), 3), dtype="uint8")
            self.kernel_fill = np.ones((kernel_fill,kernel_fill),np.uint8)

        utils.success_print('YOLO initialized')
        
    def detect(self, frame, frame_b, frame_cnt, orig_fps, starttime, set_frames):

        relevant = False
        (H, W) = frame.shape[:2]
        frame_mask = np.zeros((H, W), dtype=np.uint8)
        boxes = []
        confidences = []
        labels = []
        boxed_frame = frame.copy()

        if not g.args['gpu']:
            ln = self.net.getLayerNames()
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            layerOutputs = self.net.forward(ln)
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > g.args['confidence']:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        pts = Polygon([[x,y], [x+width,y], [x+width, y+height], [x,y+height]])
                        if g.poly_mask is None or g.poly_mask.intersects(pts):
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            labels.append(self.labels[classID])
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, g.args["confidence"], 0.3)

            if len(idxs) > 0:
                relevant = True
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = (255,0,0)
                    cv2.rectangle(boxed_frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.2f}".format(self.labels[i], confidences[i])
                
                    cv2.putText(boxed_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                
                    # add object to mask
                    delta = 5
                    d_x = max (x-delta, 0)
                    d_y = max (y-delta, 0)
                    d_w = min (W, w+delta)
                    d_h = min (H, h+delta)
                    cv2.rectangle(frame_mask, (d_x,d_y), (d_x+d_w, d_y+d_h), (255, 255, 255), cv2.FILLED)
                    obj_info = {
                        'name': self.labels[i],
                        'time':int(frame_cnt/orig_fps),
                        'frame': frame_cnt,
                        'location': ((x,y),(x+w, y+h)),
                        'confidence': '{:.4f}'.format(confidences[i])
                    }

                    text = '{}: {}s, Frame: {}'.format(self.labels[i], int(frame_cnt/orig_fps), frame_cnt)
                    if starttime:
                        st = dateparser.parse(starttime)
                        #from_time = to_time - datetime.timedelta(hours = 1)
                        # print (st)
                        dt = st + timedelta(seconds=int(frame_cnt/orig_fps))
                        text = self.labels[i] + ':' +dt.strftime('%b %d, %I:%M%p')
                        obj_info['time'] = text
                    text = text.upper()
                    if g.args['drawboxes']:
                        cv2.rectangle(frame_b, (x, y), (x + w, y + h), (255,255,255), 1)

                    utils.write_text(frame_b, text, d_x, d_y)
                    set_frames['frames'].append (obj_info)

        else:  # GPU code
            im = self.m.array_to_image(frame)
            detections = self.m.detect_image(im)
            boxes = []
            confidences = []
            labels =[]
            for detect in detections:
                (label, confidence, bbox) = detect
                if confidence > g.args['confidence']:
                    relevant = True
                    box = bbox 
                    (centerX, centerY, width, height) = box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    width = int(width)
                    height = int(height)
                    pts = Polygon([[x,y], [x+width,y], [x+width, y+height], [x,y+height]])
                    if g.poly_mask is None or g.poly_mask.intersects(pts):
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        labels.append(label) 
                        color = (255,0,0)
                        cv2.rectangle(boxed_frame, (x, y), (x + width, y + height), color, 2)
                        text = "{}: {:.2f}".format(label, confidence)
                        cv2.putText(boxed_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                        #print ('Got {} with confidence of {} at locations {} will use:{}'.format(label,confidence,bbox, confidence > g.args['confidence']))
                        # add object to mask

                        if g.args['drawboxes']:
                            cv2.rectangle(frame_b, (x, y), (x + width, y + height), (255,255,255), 1)

                        obj_info = {
                            'name': label,
                            'time':int(frame_cnt/orig_fps),
                            'frame': frame_cnt,
                            'location': ((x,y),(x+width, y+height)),
                            'confidence': '{:.4f}'.format(confidence)
                        }

                        # form text
                        text = '{}: {}s, Frame: {}'.format(label, int(frame_cnt/orig_fps), frame_cnt)
                        if starttime:
                            st = dateparser.parse(starttime)
                            #from_time = to_time - datetime.timedelta(hours = 1)
                            # print (st)
                            dt = st + timedelta(seconds=int(frame_cnt/orig_fps))
                            text = label + ':' +dt.strftime('%b %d, %I:%M%p')
                            obj_info['time'] = text
                        set_frames['frames'].append (obj_info)
                        text = text.upper()

                        # work on displaying text properly
                        delta = 5
                        d_x = max (x-delta, 0)
                        d_y = max (y-delta, 0)
                        bsx, bsy, bex, bey = utils.write_text(frame=frame_b, text=text, x=d_x, y=d_y, W=W, H=H, adjust=True)
                        cv2.rectangle(frame_mask, (bsx, bsy), (bex, bey), (255, 255, 255), cv2.FILLED)


        
        foreground_a = cv2.bitwise_and(frame,frame, mask=frame_mask)
        foreground_b = cv2.bitwise_and(frame_b,frame_b, mask=frame_mask)
      
        #combined_fg = cv2.bitwise_and(foreground_a, foreground_b)
        combined_fg= cv2.addWeighted(foreground_b, 0.5, foreground_a, 0.5,0)

        #cv2.imshow("fgb", combined_fg)
        frame_mask_inv = cv2.bitwise_not(frame_mask)

        # blend frame with foreground a missing
        modified_frame_b = cv2.bitwise_and(frame_b, frame_b, mask=frame_mask_inv)
        
       
        merged_frame = cv2.add(modified_frame_b, combined_fg)

          # draw mask on blend frame
        cv2.polylines(merged_frame, [g.raw_poly_mask], True, (0,0,255), thickness=1)

         
        #return merged_frame, foreground_a, frame_mask, relevant
        return merged_frame, foreground_a, frame_mask, relevant, boxed_frame
