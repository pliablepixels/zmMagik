import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import cv2
import numpy as np
from shapely.geometry import Polygon
import dateparser
from datetime import timedelta
from ctypes import *
import re


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class DetectYolo:

    def __init__(self,configPath=None, weightPath=None, labelsPath=None, kernel_fill=3):
        utils.success_print('Using OpenCV model for YOLO')
        utils.success_print('If you run out of memory, please tweak yolo.cfg')

        self.net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
        self.labels = open(labelsPath).read().strip().split("\n")
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.kernel_fill = np.ones((kernel_fill,kernel_fill),np.uint8)

        if g.args['gpu']:
            (maj,minor,patch) = cv2.__version__.split('.')
            min_ver = int (maj+minor)
            if min_ver < 42:
                utils.fail_print('Not setting CUDA backend for OpenCV DNN')
                utils.dim_print ('You are using OpenCV version {} which does not support CUDA for DNNs. A minimum of 4.2 is required. See https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/ on how to compile and install openCV 4.2'.format(cv2.__version__))
            else:
                utils.success_print ('Setting CUDA backend for OpenCV. If you did not set your CUDA_ARCH_BIN correctly during OpenCV compilation, you will get errors during detection related to invalid device/make_policy')
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    utils.success_print('YOLO initialized')
        
    def detect(self, frame, frame_b, frame_cnt, orig_fps, starttime, set_frames):
        relevant = False
        (H, W) = frame.shape[:2]
        frame_mask = np.zeros((H, W), dtype=np.uint8)
        boxes = []
        confidences = []
        labels = []
        boxed_frame = frame.copy()

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
                label = self.labels[classID]
                if confidence > g.args['confidence']:
                    r = re.compile(g.args['detectpattern'])
                    if not re.match(r, label):
                        #utils.dim_print('object "{}" does not match "{}"'.format(label, g.args['detectpattern']))
                        continue
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    labels.append(label)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, g.args["confidence"], 0.3)
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (width, height) = (boxes[i][2], boxes[i][3])
                label = labels[i]
                confidence = confidences[i]

                pts = Polygon([[x,y], [x+width,y], [x+width, y+height], [x,y+height]])
                if g.poly_mask is None or g.poly_mask.intersects(pts):
                    relevant = True
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    labels.append(label)
                    color = (255,0,0)
                    cv2.rectangle(boxed_frame, (x, y), (x + width, y + height), color, 2)
                    text = "{}: {:.2f}".format(label, confidence)
                    cv2.putText(boxed_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

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

                    # work on displaying text properly
                    text = text.upper()

                    delta = 0
                    d_x = max (x-delta, 0)
                    d_y = max (y-delta, 0)
                    d_w = min (W, width+delta)
                    d_h = min (H, height+delta)
                    bsx, bsy, bex, bey = utils.write_text(frame=frame_b, text=text, x=d_x, y=d_y, W=W, H=H, adjust=True)
                    # frame mask of text
                    #cv2.rectangle(frame_mask, (bsx, bsy), (bex, bey), (255, 255, 255), cv2.FILLED)
                    # frame mask of object
                    cv2.rectangle(frame_mask, (d_x,d_y), (d_x+d_w, d_y+d_h), (255, 255, 255), cv2.FILLED)

                    

        foreground_a = cv2.bitwise_and(frame,frame, mask=frame_mask)
        foreground_b = cv2.bitwise_and(frame_b,frame_b, mask=frame_mask)
        combined_fg= cv2.addWeighted(foreground_b, 0.5, foreground_a, 0.5,0)
        frame_mask_inv = cv2.bitwise_not(frame_mask)

        # blend frame with foreground a missing
        modified_frame_b = cv2.bitwise_and(frame_b, frame_b, mask=frame_mask_inv)
        merged_frame = cv2.add(modified_frame_b, combined_fg)
          # draw mask on blend frame
        cv2.polylines(merged_frame, [g.raw_poly_mask], True, (0,0,255), thickness=1)
        #return merged_frame, foreground_a, frame_mask, relevant
        return merged_frame, foreground_a, frame_mask, relevant, boxed_frame
