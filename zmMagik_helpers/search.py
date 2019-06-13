

import cv2
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import os

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.log as log


def search_video(input_file=None, out_file=None, eid = None, mid = None):
    utils.dim_print ('Analyzing: {}'.format(input_file))
    vid = cv2.VideoCapture(input_file)
    orig_fps = max(1, (g.args['fps'] or int(vid.get(cv2.CAP_PROP_FPS))))
    frame_found = False # if any match found, this will be true
    out = None

    det_type = 'found' if g.args['present'] else 'missing'
    set_frames = {
        'eventid': eid,
        'monitorid': mid,
        'type': det_type,
        'frames':[]
    }

    # if we want to write frames to a new video, 
    # make sure it uses the same FPS as the input video and is of the same size
    if g.args['write']:
        width  = int(vid.get(3))
        height = int(vid.get(4))
        if g.args['resize']:
            resize = g.args['resize']
            width = int (width * resize)
            height = int (height * resize)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h,t = os.path.split(input_file)
        h = h or '.'
        dt = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        if not out_file:
            out_file = h+'/analyzed-'+dt+'-'+t
        out = cv2.VideoWriter(out_file, fourcc, orig_fps, (width,height)) 
        print ('If frames are matched, will write to output video: {}'.format(out_file))

    # get metadata from the input video. There are times this may be off
    # fps_skip is set to 1/2 of FPS. So if you analyze a video with 10FPS, we will skip every 5 frames during analysis
    # basically, I think 2 fps for analysis is sufficient. You can override this


    if g.args['skipframes']:
        fps_skip = g.args['skipframes']
    else:
        fps_skip = max(1, int(vid.get(cv2.CAP_PROP_FPS)/2))
  
    total_frames =  int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) 
    start_time = time.time()

    utils.dim_print ('fps={}, skipping {} frames, total frames={}'.format(orig_fps, fps_skip, total_frames))
    utils.dim_print ('threshold={}, search type=if {}'.format(g.args['threshold'], det_type))
    frame_cnt = 0
    bar = tqdm(total=total_frames) 
    
    # now loop through the input video
    while True:
        succ, frame = vid.read()
        if not succ:
        
            break
        frame_cnt = frame_cnt + 1
        bar.update(1)
        if frame_cnt % fps_skip: 
            # skip frames based on our skip frames count. We don't really need to process every frame
            continue
       
        if g.args['resize']:
            resize = g.args['resize']
            rh, rw, rl = frame.shape
            frame = cv2.resize(frame, (int(rw*resize), int(rh*resize)))
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if g.args['display']:
            cv2.imshow('frame', frame_gray)
            cv2.imshow('find', g.template)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(1)

        tl,br, minv, maxv = find_in_frame(frame_gray, g.template)
        #print (maxv)
        if maxv >= g.args['threshold'] and g.args['present']:
            # if we want to record frames where the object is present
            set_frames['frames'].append ({'time': int(frame_cnt/orig_fps), 'frame':frame_cnt, 'location':(tl,br), 'accuracy':'{:.2%}'.format(maxv)})
            #matched.append('{}s, Frame: {}, at:{},{} (accuracy:{:.2%})'.format(int(frame_cnt/orig_fps),frame_cnt, tl, br, maxv))
            cv2.rectangle(frame, tl, br, (255,0,0), 2)
            if g.args['write']: 
                # put a box around the object, write to video
                text = '{}s, Frame: {}'.format(int(frame_cnt/orig_fps), frame_cnt)
        
                (tw, th) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, thickness=1)[0]
                cv2.rectangle(frame, (width-tw-5,height-th-5), (width,height), (0,0,0), cv2.FILLED)
                cv2.putText(frame, text, (width-tw-2, height-2), cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(255,255,255), thickness=1)
                out.write(frame)
            frame_found = True
            if not g.args['all']: break
        if maxv < g.args['threshold'] and not g.args['present']:
            # if we want to record frames where the object is absent
            set_frames['frames'].append ({ 'time': int(frame_cnt/orig_fps), 'frame':frame_cnt, 'location':None, 'accuracy':'{:.2%}'.format(maxv)})
            #missing.append('{}s, Frame: {} (accuracy:{:.2%})'.format(int(frame_cnt/orig_fps),frame_cnt, maxv))
            if g.args['write']: 
                text = 'MISSING: {}s, Frame: {}'.format(int(frame_cnt/orig_fps), frame_cnt)
                (tw, th) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, thickness=1)[0]
                cv2.rectangle(frame, (width-tw-5,height-th-5), (width,height), (0,0,255), cv2.FILLED)
                cv2.putText(frame, text, (width-tw-2, height-2), cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(255,255,255), thickness=1)
                out.write(frame)
            frame_found = True
            if not g.args['all']: break
        
        frame_cnt = frame_cnt+1
    # all done
    end_time = time.time()
    bar.close()
    # dump matches
    if frame_found:
        if g.args['present']:
            utils.success_print ('Match found in {} frames, starting at {}s, with initial accuracy of {}'.format(len(set_frames['frames']),set_frames['frames'][0]['time'], set_frames['frames'][0]['accuracy']))
            g.json_out.append(set_frames)
           # for match in matched:
           #    print (match)
        else:
          utils.success_print ('Object missing in {} frames, starting at {}s'.format(len(set_frames['frames']),set_frames['frames'][0]['time']))
          g.json_out.append(set_frames)
          #  for miss in missing:
          #      print (miss)
    else:
        print ('No matches found')
    if g.args['write']:
        if frame_found:
            utils.success_print ('Video of frames written  to {}'.format(out_file))
        else:
            os.remove(out_file) # blank file, no frames

    try:
        if remove_downloaded:
            os.remove(g.args['input']) # input was a remote file that was downloaded, so remove local download
    except:
        pass

    print ('\nTime: {:.2}s'.format(end_time-start_time))
    bar.close()
    vid.release()
    if out: out.release()

    return frame_found

def find_in_frame(frame, template):
    # simple template matching. This is scale invariant. For more complex
    # scale/rotation capable searching use SIFT/ORB/FLANN
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    result = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h,w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, min_val, max_val
