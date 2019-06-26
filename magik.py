'''
    To search fo objects:
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
import urllib.request
import urllib.parse as urlparse
import json
import dateparser 
import requests



if sys.version_info < (3, 0):
    sys.stdout.write("Sorry, this script requires Python 3.x, not Python 2.x\n")
    sys.exit(1)

import zmMagik_helpers.utils as utils
import zmMagik_helpers.globals as g
import zmMagik_helpers.blend as zmm_blend
import zmMagik_helpers.annotate as zmm_annotate
import zmMagik_helpers.search as zmm_search
import zmMagik_helpers.log as log


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



utils.init_colorama() # colorama

def process_timeline():

    try:
        os.remove('blended.mp4')
    except:
        pass

    url = g.args['portal']+'/api/events/index/StartTime >=:'+g.args['from']+'/EndTime <=:'+g.args['to']
    if g.args['objectonly']:
        url = url+'/Notes REGEXP:detected:'
    if g.args['alarmonly']:
        url = url+'/AlarmFrames >=:1'
    if g.args['blend'] and len(g.mon_list) > 1:
        utils.bold_print('You have chosen to blend events from multiple monitors. Results may be poor. Blending should typically be done on a fixed view (single monitor)')
    for mon in g.mon_list:
     #   print (mon)
        url = url+'/MonitorId =:'+str(mon)

    url =url+'.json?sort=StartTime&direction=asc&username='+g.args['username']+'&password='+g.args['password']
    print('Getting list of events using: {}'.format(url))
    resp = requests.get(url)
    #print (resp.json())
    events = resp.json()['events']

    
    
    cnt = 0
    delay = 0
    for event in events:
        cnt = cnt + 1
        #print (event['Event']['Id'])
        url_download= g.args['portal']+'/index.php?view=view_video&eid='+event['Event']['Id']+'&username='+g.args['username']+'&password='+g.args['password']
        in_file = url_download

        print ('\n==============| Processing Event: {} Monitor: {} ({} of {})|============='.format(event['Event']['Id'],event['Event']['MonitorId'], cnt, len(events)))

        #print ("VIDEO ID IS:",event['Event']['DefaultVideo'])
        if event['Event']['DefaultVideo'] is "":
           utils.fail_print ("ERROR: only mp4 events supported, skipping")
           continue
        if g.args['download']:
            in_file =  event['Event']['Id']+'.mp4'
            utils.dim_print ('downloading {}'.format(url_download))
            try:
                urllib.request.urlretrieve(url_download, in_file)
            except IOError as e:
                utils.fail_print('ERROR:{}'.format(e))
            except: #handle other exceptions such as attribute errors
                utils.fail_print ("Unexpected error:"+ sys.exc_info()[0])
        
        g.out_file = 'analyzed-'+event['Event']['Id']+'.mp4'
       
        #print (in_file, out_file)
        try:
            if g.args['blend']:
                res = zmm_blend.blend_video(input_file = in_file, out_file = g.out_file, eid = event['Event']['Id'],mid = event['Event']['MonitorId'], starttime=event['Event']['StartTime'], delay=delay )
                delay = delay + 2
            elif g.args['annotate']:
                res = zmm_annotate.annotate_video(input_file = in_file, out_file = g.out_file, eid = event['Event']['Id'],mid = event['Event']['MonitorId'], starttime=event['Event']['StartTime'] )
                
            elif g.args['find']:
                res = zmm_search.search_video(input_file = in_file, out_file = g.out_file, eid = event['Event']['Id'],mid = event['Event']['MonitorId'] )
            else:
                raise ValueError('Unknown mode?')
            if not g.args['all'] and res:
                break
        except IOError as e:
            utils.fail_print('ERROR:{}'.format(e))

        if g.args['download']:
            try:
                os.remove(in_file)
            except:
                pass
    





# all them arguments
ap = configargparse.ArgParser()
ap.add_argument("-c","--config",is_config_file=True, help="configuration file")
ap.add_argument("-i", "--input",  help="input video to search")
ap.add_argument("--find",  help="image to look for (needs to be same size/orientation as in video)")
ap.add_argument("--mask",  help="polygon points of interest within video being processed")
ap.add_argument("--skipframes", help="how many frames to skip", type=int)
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
ap.add_argument("--detection_type",  help="Type of detection for blending", default="background_extraction")
ap.add_argument("--config_file",  help="Config file for ML based detection with full path")
ap.add_argument("--weights_file",  help="Weights file for ML based detection with full path")
ap.add_argument("--labels_file",  help="labels file for ML based detection with full path")
ap.add_argument("--meta_file",  help="meta file for Yolo when using GPU mode")
ap.add_argument("--darknet_lib",  help="path+filename of libdarknet shared object")

ap.add_argument("--from", help = "arbitrary time range like '24 hours ago' or formal dates")
ap.add_argument("--to", help = "arbitrary time range like '2 hours ago' or formal dates")
ap.add_argument("--monitors", help = "comma separated list of monitor IDs to search")
ap.add_argument("--resize", help = "resize factor (0.5 will halve) for both matching template and video size", type=float)
ap.add_argument("--dumpjson", nargs='?',default=False,const=True, type=utils.str2bool ,help = "write analysis to JSON file")

ap.add_argument("--annotate", nargs='?', const=True,default=False, type=utils.str2bool ,help = "annotates all videos in the time range. Only applicable if using --from --to or --eventid")

ap.add_argument("--blend", nargs='?', const=True,default=False, type=utils.str2bool ,help = "overlay all videos in the time range. Only applicable if using --from --to or --eventid")

ap.add_argument("--onlyrelevant", nargs='?', const=True,default=True, type=utils.str2bool ,help = "Only write frames that have detections")


ap.add_argument("--drawboxes", nargs='?', const=True,default=False, type=utils.str2bool ,help = "draw bounding boxes aroun objects in final video")

ap.add_argument("--minblendarea",help = "minimum area in pixels to accept as object of interest in forgeground extraction. Only applicable if using--blend", type=float, default=1500)
ap.add_argument("--fontscale",help = "Size of font scale (1, 1.5 etc). Only applicable if using--blend", type=float, default=1)


ap.add_argument("--download", nargs='?',const=True,type=utils.str2bool, help = "Downloads remote videos first before analysis. Seems some openCV installations have problems with remote downloads", default=True)

ap.add_argument("--display", nargs='?',const=True,default=False,type=utils.str2bool ,help = "displays processed frames. Only applicable if using --blend")
ap.add_argument("--objectonly", nargs='?',const=True,default=False,type=utils.str2bool,help = "Only process events where objects are detected. Only applicable if using --blend")
ap.add_argument("--alarmonly", nargs='?',const=True,default=False,type=utils.str2bool ,help = "Only process events which have at least 1 alarmed frame")
ap.add_argument("--balanceintensity", nargs='?',const=True,default=False,type=utils.str2bool ,help = "If enabled, will try and match frame intensities - the darker frame will be aligned to match the brighter one. May be useful for day to night transitions, or not :-p. Works with --blend")


ap.add_argument('--present', nargs='?',default=True, const=True, type=utils.str2bool, help='look for frames where image in --match is present')
ap.add_argument('--gpu', nargs='?',default=True, const=True, type=utils.str2bool, help='enable GPU processing. Needs libdarknet.so compiled in GPU mode')

g.args = vars(ap.parse_args())
utils.process_config()

if g.args['blend']: zmm_blend.blend_init()
if g.args['annotate']: zmm_annotate.annotate_init()

utils.dim_print('-----| Arguments to be used:')
for k,v in g.args.items():
    utils.dim_print ('{}={}'.format(k,v))
print('\n')

start_time = time.time()
if g.args['from'] or g.args['to']:
    # if its a time range, ignore event/input
    process_timeline()
    
else:
    if g.args['eventid']:

        # we need to construct the url
        g.args['input'] = g.args['portal']+'/index.php?view=view_video&eid='+g.args['eventid']+'&username='+g.args['username']+'&password='+g.args['password']
        g.out_file = g.args['eventid']+'-analyzed.mp4'

    if (g.args['input'].lower().startswith(('http:', 'https:'))):
        parsed = urlparse.urlparse(g.args['input'])
        try:
            eid = urlparse.parse_qs(parsed.query)['eid'][0]
            fname = eid+'.mp4'
            g.args['eventid'] = eid
        except KeyError:
            fname = 'temp-analysis.mp4'
        if g.args['download']:
            utils.dim_print ('Downloading video from url: {}'.format(g.args['input']))
            urllib.request.urlretrieve(g.args['input'], fname)
            g.args['input'] = fname
            remove_downloaded = True
        g.out_file = 'analyzed-'+fname
       
    if g.args['find']:
        res = zmm_search.search_video(input_file=g.args['input'], out_file=g.out_file, eid=g.args['eventid'], mid=None, starttime=None, delay=0)
    elif g.args['blend']:
        res = zmm_blend.blend_video(input_file=g.args['input'], out_file=g.out_file, eid=g.args['eventid'], mid=None, starttime=None, delay=0)
    elif g.args['annotate']:
        res = zmm_annotate.annotate_video(input_file=g.args['input'],  eid=g.args['eventid'], mid=None, starttime=None)


end_time = time.time()
print ('\nTotal time: {}s'.format(round(end_time-start_time,2)))
if g.args['dumpjson']:
    jf = 'analyzed-'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.json'
    print ('Writing output to {}'.format(jf))
    with open (jf, 'w') as jo:
        json.dump(g.json_out,jo)


