
'''
    To search for objects:
        python ./magik.py  --find trash.jpg --username <user> --password <password> --portal https:/<portal>/zm --write --from "may 28 1pm" --to "may 28 5pm" --monitors 11,14 --no-present --skipframes=5
    To create a video blend within a time period:
    python ./magik.py  --username <user> --password <passwd> --portal https://<portal>/zm  --monitors <mid>   --from "jun 1, 9:58am" --to "jun 1, 10:06a"  --blend --objectonly --display  --skipframes=5 --mask="197,450 1276,463 1239,710 239,715"

The mask option filters blend matches only for that area
'''
import configargparse
import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta
import json
import dateparser
import pyzm
import pyzm.api as zmapi

if sys.version_info < (3, 0):
    sys.stdout.write("Sorry, this script requires Python 3.x, not Python 2.x\n")
    sys.exit(1)

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.blend as zmm_blend
import zmMagik_helpers.annotate as zmm_annotate
import zmMagik_helpers.search as zmm_search
import zmMagik_helpers.log as log

def remove_input(input_file=None):
    try:
        os.remove(input_file) # cleanup
    except:
        pass

def handoff(mo_id, ev_id, st_tm):
    global delay, in_file
    try:
        if g.args['blend'] and not g.args['annotate'] and not g.args['find']:
            res = zmm_blend.blend_video(input_file=in_file, out_file=g.out_file, eid=ev_id,
                                        mid=mo_id, starttime=st_tm,
                                        delay=delay)
            delay = delay + g.args['blenddelay']
        elif g.args['annotate'] and not g.args['blend'] and not g.args['find']:
            res = zmm_annotate.annotate_video(input_file=in_file, eid=ev_id,
                                              mid=mo_id,
                                              starttime=st_tm)
        elif g.args['find'] and not g.args['blend'] and not g.args['annotate']:
            res = zmm_search.search_video(input_file=in_file, out_file=g.out_file, eid=ev_id,
                                          mid=mo_id)
        else:
            raise ValueError('No support for mixing modes or you\'re trying an unknown mode')
    except IOError as e:
        utils.fail_print('ERROR: {}'.format(e))
    remove_input(in_file)

# adapted from https://stackoverflow.com/a/12117065
def float_01(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('Float {} not in range of 0.1 to 1.0'.format(x))
    return x

def float_71(x):
    x = float(x)
    if x < 0.7 or x > 1.0:
        raise argparse.ArgumentTypeError('Float {} not in range of 0.7 to 1.0'.format(x))
    return x


# colorama
utils.init_colorama()
try:
    import pyzm.ZMLog as zmlog
    zmlog.init(name='zmMagik')
except ImportError as e:
    print('Could not import ZMLog, function will be disabled:' + str(e))
    zmlog = None


# all them arguments
ap = configargparse.ArgParser()
ap.add_argument("-c","--config",is_config_file=True, help="configuration file")
ap.add_argument("-i", "--input",  help="input video to search")
ap.add_argument("--find",  help="image to look for (needs to be same size/orientation as in video)")
ap.add_argument("--mask",  help="polygon points of interest within video being processed")
ap.add_argument("--skipframes", help="how many frames to skip", type=int, default=1)
ap.add_argument("--trailframes", help="how many frames to write after relevant frame", type=int, default=10)
ap.add_argument("--blenddelay", help="how much time to wait in seconds before blending next event", type=int, default=2)
ap.add_argument("--fps", help="fps of video, to get timing correct", type=int)
ap.add_argument("--threshold", help="Only for background extraction. a number between 0 to 1 on accuracy threshold. 0.7 or above required", type=float_71, default=0.7)
ap.add_argument("--confidence", help="Only for YOLO. a number between 0 to 1 on minimum confidence score", type=float_01, default=0.6)
ap.add_argument("-a", "--all", action='store_true', help="process all frames, don't stop at first find")
ap.add_argument("-w", "--write", action='store_true', help="create video with matched frames. Only applicable for --find")
ap.add_argument("--interactive", action='store_true', help="move to next frame after keypress. Press 'c' to remove interactive")

ap.add_argument("--eventid",  help="Event id")
ap.add_argument("--username",  help="ZM username")
ap.add_argument("--password",  help="ZM password")
ap.add_argument("--portal",  help="ZM portal")
ap.add_argument("--apiportal",  help="ZM API portal")
ap.add_argument("--detection_type",  help="Type of detection for blending", default="background_extraction")
ap.add_argument("--config_file",  help="Config file for ML based detection with full path")
ap.add_argument("--weights_file",  help="Weights file for ML based detection with full path")
ap.add_argument("--labels_file",  help="labels file for ML based detection with full path")
ap.add_argument("--meta_file",  help="meta file for Yolo when using GPU mode")

ap.add_argument('--gpu', nargs='?',default=False, const=True, type=utils.str2bool, help='enable GPU processing. Needs libdarknet.so compiled in GPU mode')

ap.add_argument("--from", help = "arbitrary time range like '24 hours ago' or formal dates")
ap.add_argument("--to", help = "arbitrary time range like '2 hours ago' or formal dates")
ap.add_argument("--monitors", help = "comma separated list of monitor IDs to search")
ap.add_argument("--resize", help = "resize factor (0.5 will halve) for both matching template and video size", type=float)
ap.add_argument("--dumpjson", nargs='?',default=False,const=True, type=utils.str2bool ,help = "write analysis to JSON file")
#
ap.add_argument("--annotate", nargs='?', const=True,default=False, type=utils.str2bool ,help = "annotates all videos in the time range. Only applicable if using --from --to or --eventid")
#
ap.add_argument("--blend", nargs='?', const=True,default=False, type=utils.str2bool ,help = "overlay all videos in the time range. Only applicable if using --from --to or --eventid")
ap.add_argument("--detectpattern",  help="which objects to detect (supports regex)", default=".*")
ap.add_argument("--relevantonly", nargs='?', const=True,default=True, type=utils.str2bool ,help = "Only write frames that have detections")
#
ap.add_argument("--drawboxes", nargs='?', const=True,default=False, type=utils.str2bool ,help = "draw bounding boxes aroun objects in final video")
#
ap.add_argument("--minblendarea",help = "minimum area in pixels to accept as object of interest in forgeground extraction. Only applicable if using--blend", type=float, default=1500)
ap.add_argument("--fontscale",help = "Size of font scale (1, 1.5 etc). Only applicable if using--blend", type=float, default=1)
#
ap.add_argument("--download", nargs='?',const=True,type=utils.str2bool, help = "Downloads remote videos first before analysis. Seems some openCV installations have problems with remote downloads", default=True)
#
ap.add_argument("--display", nargs='?',const=True,default=False,type=utils.str2bool ,help = "displays processed frames. Only applicable if using --blend")
#
ap.add_argument("--show_progress", nargs='?',const=True,default=True,type=utils.str2bool ,help = "Shows progress bars")
#
ap.add_argument("--objectonly", nargs='?',const=True,default=False,type=utils.str2bool,help = "Only process events where objects are detected. Only applicable if using --blend")
ap.add_argument("--minalarmframes", help="how many alarmed frames for an event to be selected", type=int, default=None)
ap.add_argument("--maxalarmframes", help="how many alarmed frames for an event to be skipped", type=int, default=None)
ap.add_argument("--duration", help="how long (in seconds) to make the video", type=int, default=0)
#
ap.add_argument("--balanceintensity", nargs='?',const=True,default=False,type=utils.str2bool ,help = "If enabled, will try and match frame intensities - the darker frame will be aligned to match the brighter one. May be useful for day to night transitions, or not :-p. Works with --blend")
#
ap.add_argument('--present', nargs='?',default=True, const=True, type=utils.str2bool, help='look for frames where image in --match is present')
ap.add_argument('--sequential', nargs='?', default=True, const=True, type=utils.str2bool, help='Process events'
                'per monitor (i.e. you specify 2 monitors, it does the events for 1 monitor first, then the next monitor')

try:
    g.args = vars(ap.parse_args())
except Exception as e:
    print('error ConfigArgParse - {}'.format(e))
utils.process_config()

if g.args['blend']: zmm_blend.blend_init()
if g.args['annotate']: zmm_annotate.annotate_init()
utils.dim_print('-----| Arguments to be used:')
for k, v in g.args.items():
    utils.dim_print('{}={}'.format(k, v))
print('\n')

s_time = time.time()
try:
    api_options = {
        'portalurl': g.args['portal'],
        'apiurl': g.args['apiportal'],
        'user': g.args['username'],
        'password': g.args['password'],
        'logger': zmlog,  # causes an error if host doesnt have /etc/zm/zm.conf, fix in pyzm?
    }
    import traceback
    import time
    zmapi = zmapi.ZMApi(options=api_options)
except Exception as e:
    print('Error initing zmAPI: {}'.format(str(e)))
    print(traceback.format_exc())
    exit(1)

try:
    event_filter = {}
    if g.args['eventid']:
        event_filter['event_id'] = g.args['eventid']
    if g.args['from']:
        event_filter['from'] = g.args['from']
    if g.args['to']:
        event_filter['to'] = g.args['to']
    if g.args['minalarmframes']:
        event_filter['min_alarmed_frames'] = g.args['minalarmframes']
    if g.args['maxalarmframes']:
        event_filter['max_alarmed_frames'] = g.args['maxalarmframes']
    if g.args['objectonly']:
        event_filter['object_only'] = g.args['objectonly']
except Exception as e:
    print('ERROR setting event_filter keys - {}'.format(e))
    pass
ms = zmapi.monitors()
mons = g.mon_list
mon_events = {}
for m in mons:
    # create a filtered object of events for the current mID loop value
    cam_events = ms.find(id=m).events(options=event_filter)
    print('Found {} event(s) with filter: {}'.format(len(cam_events.list()), event_filter))
    cnt = 0
    # loop through the events now and extract info
    for e in cam_events.list():
        cnt += 1
        mon_events[e.id()] = [ e.id(), e.monitor_id(), e.start_time(), e.get_video_url()]

delay = 0
cnt = 0
if not g.args['sequential']:
    utils.fail_print('Mixing Events across Monitors, unexpected results may occur when blending')
    # sort list by event # (should be in correct order, if not I will try start time converted to epoch then sorted)
    sorted_mon = []
    for s_e in sorted(mon_events, reverse=True):
        sorted_mon.append(mon_events[s_e])
    #print('sorted_mon LEN ({}) = {}'.format(len(sorted_mon), sorted_mon))
    for s_event in sorted_mon:
        cnt += 1
        in_file = s_event[3]
        if g.args['download']:
            get_efs = ms.find(id=s_event[1]).events(options={'event_id': s_event[0]})
            for c in get_efs.list():
                c.download_video()
            in_file = str(s_event[0]) + '-video.mp4'
        print('\n==============| Processing Event:{} for Monitor: {} ({} of {})|============='.format(
            s_event[0], s_event[1], cnt, len(sorted_mon)))
        g.out_file = 'analyzed-mID_' + str(s_event[1]) + '-Event-' + str(s_event[0]) + '.mp4'
        handoff(s_event[1], s_event[0], s_event[2])
else:
    utils.bold_print('Sequential mode active: processing events per monitor')
    for ay in mon_events.keys():
        cnt += 1
        in_file = mon_events[ay][3]
        if g.args['download']:
            get_efs = ms.find(id=mon_events[ay][1]).events(options={'event_id': mon_events[ay][0]})
            for c in get_efs.list():
                c.download_video()
            in_file = str(mon_events[ay][0]) + '-video.mp4'
        print('\n==============| Processing Event:{} for Monitor: {} ({} of {})|============='.format(
            mon_events[ay][0], mon_events[ay][1], cnt, len(mon_events)))
        g.out_file = 'analyzed-mID_' + str(mon_events[ay][1]) + '-Event-' + str(mon_events[ay][0]) + '.mp4'
        handoff(mon_events[ay][1], mon_events[ay][0], mon_events[ay][2])
end_time = time.time()
print('\nTotal time: {}s'.format(round(end_time - s_time, 2)))
if g.args['dumpjson']:
    if g.json_out:
        jf = 'analyzed-' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.json'
        print('Writing output to {}'.format(jf))
        with open(jf, 'w') as jo:
            json.dump(g.json_out, jo)
