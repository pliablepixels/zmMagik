
import cv2
import time
from tqdm import tqdm
import os
from shapely.geometry import Polygon
import dateparser 
from datetime import datetime, timedelta
import numpy as np

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log

def blend_video(input_file=None, out_file=None, eid = None, mid = None, starttime=None):
   
    print ('Blending: {}'.format(input_file))

    vid = cv2.VideoCapture(input_file)
    orig_fps = max(1, (g.args['fps'] or int(vid.get(cv2.CAP_PROP_FPS))))
    width  = int(vid.get(3))
    height = int(vid.get(4))
    if g.args['resize']:
        resize = g.args['resize']
        print (width,height, resize)
        width = int(width * resize)
        height = int(height * resize)

    if os.path.isfile('blended.mp4'):
        vid_blend = cv2.VideoCapture('blended.mp4')
    else:
        vid_blend = None
        print ('blend file will be created in this iteration')
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outf = cv2.VideoWriter('new-blended.mp4', fourcc, orig_fps, (width,height)) 

    if g.args['skipframes']:
        fps_skip = g.args['skipframes']
    else:
        fps_skip = int(vid.get(cv2.CAP_PROP_FPS)/2)

    total_frames =  int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) 
    if vid_blend:
        total_frames = max(total_frames, int(vid_blend.get(cv2.CAP_PROP_FRAME_COUNT)) )
        utils.dim_print ('frames in eid: {} vs blend: {}'.format( int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), int(vid_blend.get(cv2.CAP_PROP_FRAME_COUNT))))

    utils.dim_print ('process {} total frames'.format(total_frames))
    start_time = time.time()
    utils.dim_print ('fps={}, skipping {} frames'.format(orig_fps, fps_skip))
    frame_cnt = 0

    bar = tqdm(total=total_frames)

   
    #fgbg = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=50, detectShadows=True)
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:
        succ, frame = vid.read()
        relevant = True
        frame_cnt = frame_cnt + 1
        bar.update(1) 
        if frame_cnt % fps_skip: 
            # skip frames based on our skip frames count. We don't really need to process every frame
            continue
        # skip fps only applies to input image. Blend video will always have fps skip applied

        if succ and g.args['resize']:
            resize = g.args['resize']
            rh, rw, rl = frame.shape
            frame = cv2.resize(frame, (int(rw*resize), int(rh*resize)))
        succ_b = False
        if vid_blend: succ_b, frame_b = vid_blend.read()
       # print (succ, succ_b)
        if not succ and not succ_b:
            utils.dim_print ('both videos are done')
            break
       
        # now populate frame and frame_b correctly:

        if (succ and not vid_blend):
            frame_b = frame.copy()

        elif (not succ and succ_b):
            frame = frame_b.copy()
        
        elif (not succ_b and succ):
            frame_b = frame.copy()

     
        relevant = False

        frame_mask = g.fgbg.apply(frame)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, g.kernel_clean)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, g.kernel_fill)
        
        # don't need this as shadows are off
        # remove grey areas
        #indices = frame_mask > 100
        #frame_mask[indices] = 255
        # get only foreground images from the new frame
        foreground_a = cv2.bitwise_and(frame,frame, mask=frame_mask)
        # clear out parts on blended frames where forground will be added
        frame_mask_inv = cv2.bitwise_not(frame_mask)
        #print (frame_mask_inv)
        modified_frame_b = cv2.bitwise_and(frame_b, frame_b, mask=frame_mask_inv)

        ctrs,_ =  cv2.findContours(frame_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        merged_frame = cv2.add(modified_frame_b, foreground_a)
        relevant = False;
        for c in ctrs:
            area = cv2.contourArea(c)
            if area > 2500:
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
                    (tw, th) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, thickness=2)[0]

                    loc_x1 = x
                    loc_y1 = y - th - 4
                    loc_x2 = x + tw + 4
                    loc_y2 = y


                    cv2.rectangle(merged_frame, (loc_x1, loc_y1), (loc_x1+tw+4,loc_y1+th+4), (0,0,0), cv2.FILLED)
                    cv2.putText(merged_frame, text, (loc_x1+2, loc_y2-2), cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=(255,255,255), thickness=1)
                    #cv2.rectangle(merged_frame,(x,y),(x+w,y+h),(0,255,0),2)


        if relevant:
            if g.args['display']:
                x = 640
                y = 480
                r_frame_b = cv2.resize (frame_b, (x, y))
                r_frame = cv2.resize (frame, (x,y))
                r_fga = cv2.resize (foreground_a, (x,y))
                r_frame_mask = cv2.resize (frame_mask, (x, y))
                r_frame_mask = cv2.cvtColor(r_frame_mask, cv2.COLOR_GRAY2BGR)
                r_merged_frame = cv2.resize (merged_frame, (x, y))
                h1 = np.hstack((r_frame, r_frame_b))
                h2 = np.hstack((r_frame_mask, r_merged_frame))
                f = np.vstack((h1,h2))
                cv2.imshow('display', f)
                #cv2.imshow('merged_frame',cv2.resize(merged_frame, (640,480)))
                #cv2.imshow('frame_mask',cv2.resize(frame_mask, (640,480)))

                #cv2.imshow('frame_mask',frame_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit(1)
            outf.write(merged_frame)
        else:
            #print ('irrelevant frame {}'.format(frame_cnt))
            pass
       

    try:
        os.remove('blended.mp4')
    except:
       pass

    bar.close()
   
    vid.release()
    outf.release()
    if vid_blend: vid_blend.release() 
    os.rename ('new-blended.mp4', 'blended.mp4')
    return False
