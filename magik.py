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
import zmMagik_helpers.search as zmm_search
import zmMagik_helpers.log as log


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
    for mon in g.mon_list:
     #   print (mon)
        url = url+'/MonitorId =:'+str(mon)
    url =url+'.json?sort=StartTime&direction=asc&username='+g.args['username']+'&password='+g.args['password']
    print('Getting list of events using: {}'.format(url))
    resp = requests.get(url)
    #print (resp.json())
    events = resp.json()['events']

    
    start_time = time.time()
    cnt = 0
    delay = 0
    for event in events:
        cnt = cnt + 1
        #print (event['Event']['Id'])
        url_download= g.args['portal']+'/index.php?view=view_video&eid='+event['Event']['Id']+'&username='+g.args['username']+'&password='+g.args['password']
        in_file = url_download
        #in_file =  event['Event']['Id']+'.mp4'
        #utils.dim_print ('downloading {}'.format(url_download))
        #urllib.request.urlretrieve(url_download, in_file)
        g.out_file = 'analyzed-'+event['Event']['Id']+'.mp4'
        print ('\n==============| Processing Event: {} Monitor: {} ({} of {})|============='.format(event['Event']['Id'],event['Event']['MonitorId'], cnt, len(events)))
        #print (in_file, out_file)
        if g.args['blend']:
            res = zmm_blend.blend_video(input_file = in_file, out_file = g.out_file, eid = event['Event']['Id'],mid = event['Event']['MonitorId'], starttime=event['Event']['StartTime'], delay=delay )
            delay = delay + 1
        else:
            res = zmm_search.search_video(input_file = in_file, out_file = g.out_file, eid = event['Event']['Id'],mid = event['Event']['MonitorId'] )
        if not g.args['all'] and res:
            break
    end_time = time.time()
    print ('\nTotal time: {:.2}s'.format(end_time-start_time))





# all them arguments
ap = configargparse.ArgParser()
ap.add_argument("-c","--config",is_config_file=True, help="configuration file")
ap.add_argument("-i", "--input",  help="input video to search")
ap.add_argument("--find",  help="image to look for (needs to be same size/orientation as in video)")
ap.add_argument("--mask",  help="polygon points of interest within video being processed")
ap.add_argument("--skipframes", help="how many frames to skip", type=int)
ap.add_argument("--fps", help="fps of video, to get timing correct", type=int)
ap.add_argument("--threshold", help="a number between 0 to 1 on accuracy threshold. 0.7 or above required", type=float, default=0.7, choices=(0.7, 1.0))
ap.add_argument("-a", "--all", action='store_true', help="process all frames, don't stop at first find")
ap.add_argument("-w", "--write", action='store_true', help="create video with matched frames")
ap.add_argument("--eventid",  help="Event id")
ap.add_argument("--username",  help="ZM username")
ap.add_argument("--password",  help="ZM password")
ap.add_argument("--portal",  help="ZM portal")
ap.add_argument("--from", help = "arbitrary time range like '24 hours ago' or formal dates")
ap.add_argument("--to", help = "arbitrary time range like '2 hours ago' or formal dates")
ap.add_argument("--monitors", help = "comma separated list of monitor IDs to search")
ap.add_argument("--resize", help = "resize factor (0.5 will halve) for both matching template and video size", type=float)
ap.add_argument("--dumpjson", action='store_true' ,help = "write analysis to JSON file")
ap.add_argument("--blend", action='store_true' ,help = "overlay all videos in the time range. Only applicable if using --from --to")
ap.add_argument("--minblendarea",help = "minimum area in pixels to accept as object of interest in forgeground extraction. Only applicable if using--blend", type=float, default=1500)
ap.add_argument("--fontscale",help = "Size of font scale (1, 1.5 etc). Only applicable if using--blend", type=float, default=1)

ap.add_argument("--display", action='store_true' ,help = "displays processed frames. Only applicable if using --blend")
ap.add_argument("--objectonly", action='store_true' ,help = "Only process events where objects are detected. Only applicable if using --blend")
ap.add_argument("--alarmonly", action='store_true' ,help = "Only process events which have at least 1 alarmed frame")
ap.add_argument("--balanceintensity", action='store_true' ,help = "If enabled, will try and match frame intensities - the darker frame will be aligned to match the brighter one. May be useful for day to night transitions, or not :-p. Works with --blend")


ap.add_argument('--present', dest='present', action='store_true', help='look for frames where image in --match is present')
ap.add_argument('--not-present', dest='present', action='store_false', help='look for frames where image in --match is NOT present')
ap.add_argument('--no-present', dest='present', action='store_false', help='look for frames where image in --match is NOT present')
ap.set_defaults(present=True)
g.args = vars(ap.parse_args())
utils.process_config()

utils.dim_print('-----| Arguments to be used:')
for k,v in g.args.items():
    utils.dim_print ('{}={}'.format(k,v))
print('\n')
if g.args['from'] or g.args['to']:
    # if its a time range, ignore event/input
    process_timeline()
    
else:
    if g.args['eventid']:
        # we need to construct the url
        g.args['input'] = g.args['portal']+'/index.php?view=view_video&eid='+g.args['eventid']+'&username='+g.args['username']+'&password='+g.args['password']
        g.out_file = g.args['eventid']+'-analyzed.mp4'

    if (g.args['input'].lower().startswith(('http:', 'https:'))):
        dim_print ('Downloading video from url: {}'.format(g.args['input']))
        parsed = urlparse.urlparse(g.args['input'])
        try:
            eid = urlparse.parse_qs(parsed.query)['eid'][0]
            fname = eid+'.mp4'
            g.args['eventid'] = eid
        except KeyError:
            fname = 'temp-analysis.mp4'

        urllib.request.urlretrieve(g.args['input'], fname)
        g.args['input'] = fname
        g.out_file = 'analyzed-'+g.args['input']
        remove_downloaded = True
    
    zmm_analyze.analyze_video(input_file=g.args['input'], out_file=g.out_file, eid=g.args['eventid'])

if g.args['dumpjson']:
    jf = 'analyzed-'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.json'
    print ('Writing output to {}'.format(jf))
    with open (jf, 'w') as jo:
        json.dump(g.json_out,jo)


