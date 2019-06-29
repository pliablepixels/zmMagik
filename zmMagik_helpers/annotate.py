
# FIXME - most of this code is really blend code too
# need to combine them better

import cv2
import time
from tqdm import tqdm
import os
import numpy as np

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log
from datetime import datetime

import imutils

import zmMagik_helpers.FVS as FVS
from imutils.video import FPS


det = None
det2 = None
annotate_filename = 'annotated-'
if len(g.mon_list) == 1:
    annotate_filename = annotate_filename +'mon-'+ g.mon_list[0] + '-'
annotate_filename = annotate_filename+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.mp4'

def annotate_init():
    global det, det2
    #print (g.args['detection_type'])
    if g.args['detection_type'] == 'background_extraction':
        import zmMagik_helpers.detect_background as FgBg
        det = FgBg.DetectBackground(min_accuracy = g.args['threshold'], min_blend_area=g.args['minblendarea'])

    elif g.args['detection_type'] == 'yolo_extraction':
        import zmMagik_helpers.detect_yolo as Yolo
        det = Yolo.DetectYolo (configPath = g.args['config_file'],
                              weightPath = g.args['weights_file'],
                              labelsPath =  g.args['labels_file'],
                              darknetLib = g.args['darknet_lib'])
    

    elif g.args['detection_type'] == 'mixed':
        import zmMagik_helpers.detect_background as FgBg
        import zmMagik_helpers.detect_yolo as Yolo
        det =  FgBg.DetectBackground(min_accuracy = g.args['threshold'], min_blend_area=g.args['minblendarea'])
        det2 = Yolo.DetectYolo (configPath = g.args['config_file'],
                              weightPath = g.args['weights_file'],
                              darknetLib = g.args['darknet_lib'],
                              labelsPath =  g.args['labels_file'] )
 

    else:
        raise ValueError ('Detection type {} is not known'.format(g.args['detection_type']))

    utils.bold_print('Detection mode is: {}'.format(g.args['detection_type']))

def annotate_video(input_file=None,  eid = None, mid = None, starttime=None):

    global det, det2

    set_frames = {
        'eventid': eid,
        'monitorid': mid,
        'type': 'object',
        'frames':[]
        }

    print ('annotating: {}'.format(utils.secure_string(input_file)))
    
    #vid = cv2.VideoCapture(input_file)
    vid = FVS.FileVideoStream(input_file)
    time.sleep(1)
    cvobj = vid.get_stream_object()
    vid.start()
    if not cvobj.isOpened(): 
        raise ValueError('Error reading video {}'.format(utils.secure_string(input_file)))

    if not g.orig_fps:
        orig_fps = max(1, (g.args['fps'] or int(cvobj.get(cv2.CAP_PROP_FPS))))
        g.orig_fps = orig_fps
    else:
        orig_fps = g.orig_fps

    
    width  = int(cvobj.get(3))
    height = int(cvobj.get(4))
    
    if g.args['resize']:
        resize = g.args['resize']
       # print (width,height, resize)
        width = int(width * resize)
        height = int(height * resize)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outf = cv2.VideoWriter(annotate_filename, fourcc, orig_fps, (width,height), True) 
    utils.bold_print('Output video will be {}px*{}px @ {}fps'.format(width, height, orig_fps))

    if g.args['skipframes']:
        fps_skip = g.args['skipframes']
    else:
        fps_skip = max(1,int(cvobj.get(cv2.CAP_PROP_FPS)/2))


    total_frames =  int(cvobj.get(cv2.CAP_PROP_FRAME_COUNT)) 
   

    start_time = time.time()
    utils.dim_print ('fps={}, skipping {} frames'.format(orig_fps, fps_skip))
    

    bar_annotate_video = tqdm (total=total_frames, desc='annotating')

    frame_cnt = 0
    while True:
        if vid.more():
            frame = vid.read()
            if frame is None:
                succ = False
            else:
                succ = True
        else:
            frame = None
            succ = False
        #succ, frame = vid.read()
        if not succ: break
       
        frame_cnt = frame_cnt + 1

        if not frame_cnt % 10:
            bar_annotate_video.update(10)
            

        if frame_cnt % fps_skip:
            continue
      
        if succ and g.args['resize']:
            resize = g.args['resize']
            rh, rw, rl = frame.shape
            frame = cv2.resize(frame, (int(rw*resize), int(rh*resize)))

        frame_b = frame.copy()
        merged_frame, foreground_a, frame_mask, relevant, boxed_frame = det.detect(frame, frame_b, frame_cnt, orig_fps, starttime, set_frames)
        if relevant and g.args['detection_type'] == 'mixed':
            bar_annotate_video.set_description('YOLO running')
            #utils.dim_print('Adding YOLO, found relevance in backgroud motion')
            merged_frame, foreground_a, frame_mask, relevant, boxed_frame = det2.detect(frame, frame_b, frame_cnt, orig_fps, starttime, set_frames)  
            bar_annotate_video.set_description('annotating')        
      
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

        if relevant or not g.args['relevantonly']:
            #print ("WRITING FRAME")
            outf.write (merged_frame)

        else:
            #print ('irrelevant frame {}'.format(frame_cnt))
            pass
        
   

    bar_annotate_video.close()
    vid.stop()
    outf.release()

    utils.success_print('annotated file updated in {}'.format(annotate_filename))

    g.json_out.append(set_frames)

    return False
