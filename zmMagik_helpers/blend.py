
import cv2
import time
from tqdm import tqdm
import os
import numpy as np

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log
import zmMagik_helpers.detect_background as det_bk

def blend_video(input_file=None, out_file=None, eid = None, mid = None, starttime=None, delay=0):
   
    print ('Blending: {}'.format(input_file))

    det = det_bk.DetectBackground()
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
        #utils.dim_print('Video blend {}'.format(vid_blend))
    else:
        vid_blend = None
        print ('blend file will be created in this iteration')
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outf = cv2.VideoWriter('new-blended.mp4', fourcc, orig_fps, (width,height)) 

    if g.args['skipframes']:
        fps_skip = g.args['skipframes']
    else:
        fps_skip = max(1,int(vid.get(cv2.CAP_PROP_FPS)/2))

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

        if (orig_fps * frame_cnt >= delay):
            succ, frame = vid.read()
        else:
            succ = False
            frame = None
            utils.dim_print ('waiting for {}s'.format(delay))
        
        succ_b = False
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

       
        if vid_blend: succ_b, frame_b = vid_blend.read()
        #print (succ, succ_b)
        if not succ and not succ_b:
            utils.dim_print ('both videos are done')
            break
       
        # now populate frame and frame_b correctly:
        analyze = True

        if (succ and not succ_b):
           # print ('blend over')
            frame_b = frame.copy()

        elif (not succ and succ_b):
            #print ('new video over')
            frame = frame_b.copy()
            merged_frame = frame
            foreground_a = frame
            #frame_mask = frame
            #print (frame.shape)
            frame_mask = np.ones(frame.shape,dtype=np.uint8)
            # if we are only left with past blends, just write it
            analyze = False
            relevant = True

        # draw mask on blend frame
        cv2.polylines(frame_b, [g.raw_poly_mask], True, (0,0,255), thickness=1)

        if analyze:
            
            merged_frame, foreground_a, frame_mask, relevant = det.detect(frame, frame_b, frame_cnt, orig_fps, starttime)
            
            # don't need this as shadows are off
            # remove grey areas
            #indices = frame_mask > 100
            #frame_mask[indices] = 255
            # get only foreground images from the new frame
            

        if g.args['display']:
                x = 320
                y = 240
                r_frame_b = cv2.resize (frame_b, (x, y))
                r_frame = cv2.resize (frame, (x,y))
                r_fga = cv2.resize (foreground_a, (x,y))
                r_frame_mask = cv2.resize (frame_mask, (x, y))
                try:
                    r_frame_mask = cv2.cvtColor(r_frame_mask, cv2.COLOR_GRAY2BGR)
                except:
                    pass
                r_merged_frame = cv2.resize (merged_frame, (x, y))
                h1 = np.hstack((r_frame, r_frame_mask))
                h2 = np.hstack((r_fga, r_merged_frame))
                f = np.vstack((h1,h2))
                cv2.imshow('display', f)
                #cv2.imshow('merged_frame',cv2.resize(merged_frame, (640,480)))
                #cv2.imshow('frame_mask',cv2.resize(frame_mask, (640,480)))

                #cv2.imshow('frame_mask',frame_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit(1)

        # we write if either new frame has foreground, or there is a blend to write
        if relevant or succ_b:
            outf.write(merged_frame)

        else:
            #print ('irrelevant frame {}'.format(frame_cnt))
            pass
        
   

    bar.close()
    vid.release()
    outf.release()
    if vid_blend: vid_blend.release() 
    #input("Press Enter to continue...")
    try:
        os.remove('blended.mp4')
    except:
       pass
    os.rename ('new-blended.mp4', 'blended.mp4')
    return False
