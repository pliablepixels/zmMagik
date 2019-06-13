
import cv2
import time
from tqdm import tqdm
import os
import numpy as np

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log
from datetime import datetime

det = None
blend_filename = 'blended-'
if len(g.mon_list) == 1:
    blend_filename = blend_filename +'mon-'+ g.mon_list[0] + '-'
blend_filename = blend_filename+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.mp4'

def blend_init():
    global det
    #print (g.args['detection_type'])
    if g.args['detection_type'] == 'background_extraction':
        import zmMagik_helpers.detect_background as det
        det = det.DetectBackground(min_accuracy = g.args['threshold'], min_blend_area=g.args['minblendarea'])

    elif g.args['detection_type'] == 'yolo_extraction':
        import zmMagik_helpers.detect_yolo as det
        det = det.DetectYolo (configPath = g.args['config_file'],
                              weightsPath = g.args['weights_file'],
                              labelsPath =  g.args['labels_file'] )
    

    else:
        raise ValueError ('Detection type {} is not known'.format(g.args['detection_type']))

def blend_video(input_file=None, out_file=None, eid = None, mid = None, starttime=None, delay=0):
    global det
    print ('Blending: {}'.format(input_file))
    
    vid = cv2.VideoCapture(input_file)
    if not vid.isOpened(): 
        raise ValueError('Error reading video {}'.format(input_file))

    if not g.orig_fps:
        orig_fps = max(1, (g.args['fps'] or int(vid.get(cv2.CAP_PROP_FPS))))
        g.orig_fps = orig_fps
    else:
        orig_fps = g.orig_fps

    width  = int(vid.get(3))
    height = int(vid.get(4))
    if g.args['resize']:
        resize = g.args['resize']
       # print (width,height, resize)
        width = int(width * resize)
        height = int(height * resize)

    if os.path.isfile(blend_filename):
        vid_blend = cv2.VideoCapture(blend_filename)
        #utils.dim_print('Video blend {}'.format(vid_blend))
    else:
        vid_blend = None
        print ('blend file will be created in this iteration')
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outf = cv2.VideoWriter('new-blended-temp.mp4', fourcc, orig_fps, (width,height)) 
    utils.bold_print('Output video will be {}fps'.format(orig_fps))

    if g.args['skipframes']:
        fps_skip = g.args['skipframes']
    else:
        fps_skip = max(1,int(vid.get(cv2.CAP_PROP_FPS)/2))


    total_frames_vid =  int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) 
    total_frames_vid_blend = 0

    if vid_blend:
        total_frames_vid_blend =int(vid_blend.get(cv2.CAP_PROP_FRAME_COUNT)) 
        utils.dim_print ('frames in new video: {} vs blend: {}'.format( total_frames_vid, total_frames_vid_blend))

    start_time = time.time()
    utils.dim_print ('fps={}, skipping {} frames'.format(orig_fps, fps_skip))
    utils.dim_print ('delay for new video is {}s'.format(delay))

    bar_new_video = tqdm(total=total_frames_vid, desc='New video')
    bar_blend_video = tqdm (total=total_frames_vid_blend, desc='Blend')

    
    # first wait for delay seconds
    # will only come in if blend video exists, as in first iter it is 0
    if delay:
        frame_cnt = 0
        bar_new_video.set_description ('waiting for {}s'.format(delay))
        prev_good_frame_b = None
        while True:
            if vid_blend is None:
                frame_b = prev_good_frame_b
            else:
                succ_b, frame_b = vid_blend.read()
                if not  succ_b: 
                    vid_blend = None
                else:
                    prev_good_frame_b = frame_b
            frame_cnt = frame_cnt + 1
            bar_blend_video.update(1)
            outf.write(frame_b)
            if (delay * orig_fps < frame_cnt):
           # if (frame_cnt/orig_fps > delay):
                #utils.dim_print('wait over')
              #  print ('DELAY={} ORIGFPS={} FRAMECNT={}'.format(delay, orig_fps, frame_cnt))
                break
              
    # now read new video along with blend
    bar_new_video.set_description ('New video')
    frame_cnt = 0
    while True:
        succ, frame = vid.read()
        
        frame_cnt = frame_cnt + 1
        bar_new_video.update(1)

        if frame_cnt % fps_skip:
            continue
      
        succ_b = False
        if succ and g.args['resize']:
            resize = g.args['resize']
            rh, rw, rl = frame.shape
            frame = cv2.resize(frame, (int(rw*resize), int(rh*resize)))

        if vid_blend: 
            succ_b, frame_b = vid_blend.read()
            if succ_b:
                bar_blend_video.update(1)
            else:
                vid_blend = None
            # frame_b is always resized

        if not succ and not succ_b:
                utils.dim_print ('both videos are done')
                break
       
        elif succ and succ_b:
            analyze = True
            relevant = False # may change on analysis
            #print ("succ and succ_b")

        elif succ and not succ_b:
           # print ('blend over')
            frame_b = frame.copy()
            analyze = True
            relevant = False # may change on analysis
            #print ("succ and not succ_b")

        elif not succ and succ_b:
            merged_frame = frame_b
            frame = frame_b
            boxed_frame = np.zeros_like(frame_b)
            txh,txw,_ = frame_b.shape
            frame_mask= np.zeros((txh, txw),dtype=np.uint8)
            foreground_a = np.zeros_like(frame_b)
            analyze = False
            relevant = True
            #print ("not succ and succ_b")
        
        if analyze:
            # only if both blend and new were read
            if g.args['balanceintensity']:
                intensity = np.mean(frame)
                intensity_b = np.mean(frame_b)
                if intensity > intensity_b:
                    # new frame is brighter
                    frame_b = utils.hist_match(frame_b, frame) 
                else:
                    # blend is brighter
                    frame = utils.hist_match(frame, frame_b)     
               
            merged_frame, foreground_a, frame_mask, relevant, boxed_frame = det.detect(frame, frame_b, frame_cnt, orig_fps, starttime)
      
        if g.args['display']:
                x = 320
                y = 240
                r_frame_b = cv2.resize (frame_b, (x, y))
                r_frame = cv2.resize (boxed_frame, (x,y))
                r_fga = cv2.resize (foreground_a, (x,y))
                r_frame_mask = cv2.resize (frame_mask, (x, y))
                r_frame_mask = cv2.cvtColor(r_frame_mask, cv2.COLOR_GRAY2BGR)
                r_merged_frame = cv2.resize (merged_frame, (x, y))
                h1 = np.hstack((r_frame, r_frame_mask))
                h2 = np.hstack((r_fga, r_merged_frame))
                f = np.vstack((h1,h2))
                cv2.imshow('display', f)
                #cv2.imshow('merged_frame',cv2.resize(merged_frame, (640,480)))
                #cv2.imshow('frame_mask',cv2.resize(frame_mask, (640,480)))

                #cv2.imshow('frame_mask',frame_mask)
                
                    

        if g.args['interactive']:
            key = cv2.waitKey(0)
        
        else:
            key = cv2.waitKey(1)
        if key& 0xFF == ord('q'):
            exit(1)
        if key& 0xFF == ord('c'):
            g.args['interactive']=False

        if relevant:
            outf.write(merged_frame)

        else:
            #print ('irrelevant frame {}'.format(frame_cnt))
            pass
        
   

    bar_blend_video.close()
    bar_new_video.close()
    vid.release()
    outf.release()
    if vid_blend: vid_blend.release() 
    #input("Press Enter to continue...")
    try:
        os.remove(blended_filename)
    except:
       pass
    os.rename ('new-blended-temp.mp4', blend_filename)
    utils.success_print('Blended file updated in {}'.format(blend_filename))
    return False
