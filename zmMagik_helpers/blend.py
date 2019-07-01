
import cv2
import time
from tqdm import tqdm
import os
import numpy as np

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log
from datetime import datetime

import zmMagik_helpers.FVS as FVS
from imutils.video import FPS

det = None
det2 = None
blend_filename = 'blended-'
if len(g.mon_list) == 1:
    blend_filename = blend_filename +'mon-'+ g.mon_list[0] + '-'
blend_filename = blend_filename+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.mp4'

def blend_init():
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

def blend_video(input_file=None, out_file=None, eid = None, mid = None, starttime=None, delay=0):

    global det, det2
    create_blend = False
    blend_frame_written_count = 0

    set_frames = {
        'eventid': eid,
        'monitorid': mid,
        'type': 'object',
        'frames':[]
        }

    print ('Blending: {}'.format(utils.secure_string(input_file)))
    
    vid = FVS.FileVideoStream(input_file)
    time.sleep(1)
    #vid = cv2.VideoCapture(input_file)
    cvobj = vid.get_stream_object()
    if not cvobj.isOpened(): 
        raise ValueError('Error reading video {}'.format(utils.secure_string(input_file)))

    total_frames_vid =  int(cvobj.get(cv2.CAP_PROP_FRAME_COUNT)) 
    vid.start()
    
    if not g.orig_fps:
        orig_fps = max(1, (g.args['fps'] or int(cvobj.get(cv2.CAP_PROP_FPS))))
        g.orig_fps = orig_fps
    else:
        orig_fps = g.orig_fps

    width  = int(cvobj.get(3))
    height = int(cvobj.get(4))
    if g.args['resize']:
        resize = g.args['resize']
        #print (width,height, resize)
        width = int(width * resize)
        height = int(height * resize)

    total_frames_vid_blend = 0
    if os.path.isfile(blend_filename):
        vid_blend = FVS.FileVideoStream(blend_filename)
        time.sleep(1)
        cvobj_blend = vid_blend.get_stream_object()
        total_frames_vid_blend =int(cvobj_blend.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_blend.start()
        #vid_blend = cv2.VideoCapture(blend_filename)
        #utils.dim_print('Video blend {}'.format(vid_blend))
    else:
        vid_blend = None
        cvobj_blend = None
        print ('blend file will be created in this iteration')
        create_blend = True
    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outf = cv2.VideoWriter('new-blended-temp.mp4', fourcc, orig_fps, (width,height)) 
    utils.bold_print('Output video will be {}*{}@{}fps'.format(width,height,orig_fps))

    if g.args['skipframes']:
        fps_skip = g.args['skipframes']
    else:
        fps_skip = max(1,int(cvobj.get(cv2.CAP_PROP_FPS)/2))


    if vid_blend: 
        utils.dim_print ('frames in new video: {} vs blend: {}'.format( total_frames_vid, total_frames_vid_blend))

    start_time = time.time()
    utils.dim_print ('fps={}, skipping {} frames'.format(orig_fps, fps_skip))
    utils.dim_print ('delay for new video is {}s'.format(delay))

    bar_new_video = tqdm(total=total_frames_vid, desc='New video', miniters = 10)
    bar_blend_video = tqdm (total=total_frames_vid_blend, desc='Blend', miniters = 10)

    is_trailing = False
    blend_frames_read = 0
    # first wait for delay seconds
    # will only come in if blend video exists, as in first iter it is 0
    # However, if blend wasn't created (no relevant frames), ignore delay
    if delay and not create_blend:
        frame_cnt = 0
        bar_new_video.set_description ('waiting for {}s'.format(delay))
        prev_good_frame_b = None
        a = 0
        b = 0
        while True:
            if vid_blend and vid_blend.more():
                frame_b = vid_blend.read()
                if frame_b is None:
                    succ_b = False
                else:
                    succ_b = True
                    a = a + 1
                    #print ('delay read: {}'.format(a))
                    blend_frames_read = blend_frames_read + 1
                    prev_good_frame_b = frame_b
            else:
                succ_b = False
                vid_blend = None
            
            # If we have reached the end of blend, but have a good last frame
            # lets use it
            if not succ_b and prev_good_frame_b is not None:
                frame_b = prev_good_frame_b
                succ_b = True
            
            if not succ_b and not prev_good_frame_b:
                break

            frame_cnt = frame_cnt + 1
            bar_blend_video.update(1)
            outf.write(frame_b)
            frame_dummy= np.zeros_like(frame_b)
            if g.args['display']:
                x = 320
                y = 240
                r_frame_b = cv2.resize (frame_b, (x, y))
                r_frame_dummy = cv2.resize (frame_dummy, (x,y))
            
                h1 = np.hstack((r_frame_dummy, r_frame_dummy))
                h2 = np.hstack((r_frame_dummy, r_frame_b))
                f = np.vstack((h1,h2))
                cv2.imshow('display', f)
 
            if g.args['interactive']:
                key = cv2.waitKey(0)
            
            else:
                key = cv2.waitKey(1)
            if key& 0xFF == ord('q'):
                exit(1)
            if key& 0xFF == ord('c'):
                g.args['interactive']=False

            blend_frame_written_count = blend_frame_written_count + 1
            b = b + 1
            #print ('delay write: {}'.format(b))
            if (delay * orig_fps < frame_cnt):
        # if (frame_cnt/orig_fps > delay):
                #utils.dim_print('wait over')
            #  print ('DELAY={} ORIGFPS={} FRAMECNT={}'.format(delay, orig_fps, frame_cnt))
                break
              
    # now read new video along with blend
    bar_new_video.set_description ('New video')
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
        
        frame_cnt = frame_cnt + 1
        bar_new_video.update(1)

        if frame_cnt % fps_skip:
            continue
      
        if succ and g.args['resize']:
            resize = g.args['resize']
            rh, rw, rl = frame.shape
            frame = cv2.resize(frame, (int(rw*resize), int(rh*resize)))

        succ_b = False
        if vid_blend: 
            if vid_blend.more():
                frame_b = vid_blend.read()
                if frame_b is None:
                    succ_b = False
                else:
                    succ_b = True
                    bar_blend_video.update(1)
                    blend_frames_read = blend_frames_read + 1
           
        if not succ and not succ_b:
                bar_blend_video.write ('both videos are done')
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
            #h1, w1 = frame.shape[:2]
            #hm, wm = frame_b.shape[:2]
        
            #print ("{}*{} frame == {}*{} frame_b".format(h1,w1,hm,wm))
            merged_frame, foreground_a, frame_mask, relevant, boxed_frame = det.detect(frame, frame_b, frame_cnt, orig_fps, starttime, set_frames)
            #print ('RELEVANT={}'.format(relevant))
            if relevant and g.args['detection_type'] == 'mixed':
                bar_new_video.set_description('YOLO running')
                #utils.dim_print('Adding YOLO, found relevance in backgroud motion')
                merged_frame, foreground_a, frame_mask, relevant, boxed_frame = det2.detect(frame, frame_b, frame_cnt, orig_fps, starttime, set_frames)  
                #print ('YOLO RELEVANT={}'.format(relevant))
                bar_new_video.set_description('New video')        
      
            if relevant:
                is_trailing = True
                trail_frames = 0

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

        # if we read a blend frame, merged frame will always be written
        # if we don't have a blend frame, then we write new frame only if its relevant
        # assuming we want relevant frames
        if relevant or not g.args['relevantonly'] or succ_b:
            #print ("WRITING")
            outf.write(merged_frame)
            blend_frame_written_count = blend_frame_written_count + 1
        elif is_trailing:
            trail_frames = trail_frames + 1

            if trail_frames > g.args['trailframes']: 
                start_trailing = False
            else:
                bar_new_video.set_description('Trailing frame')
               # bar_new_video.write('trail frame: {}'.format(trail_frames))
                outf.write(merged_frame)
                blend_frame_written_count = blend_frame_written_count + 1
        else:
            #print ('irrelevant frame {}'.format(frame_cnt))
            pass
        
   

    bar_blend_video.close()
    bar_new_video.close()
    vid.stop()
    outf.release()
    if vid_blend: vid_blend.stop() 
    print('\n')
    #input("Press Enter to continue...")
    if create_blend and blend_frame_written_count == 0:
        utils.fail_print('No relevant frames found, blend file not created. Will try next iteration')
        os.remove('new-blended-temp.mp4')
    else:
        rel = 'relevant ' if g.args['relevantonly'] else ''
        utils.success_print ('{} total {}frames written to blend file ({} read)'.format(blend_frame_written_count, rel, blend_frames_read))
        if blend_frame_written_count:
            try:
                os.remove(blend_filename)
            except:
                pass
            os.rename ('new-blended-temp.mp4', blend_filename)
            utils.success_print('Blended file updated in {}'.format(blend_filename))
        else:
            utils.success_print ('No frames written this round, not updating blend file')
        g.json_out.append(set_frames)

    return False
