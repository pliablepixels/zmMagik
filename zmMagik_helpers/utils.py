from colorama import init, Fore, Style
from shapely.geometry import Polygon
import cv2
import dateparser
import configargparse
import numpy as np


import zmMagik_helpers.globals as g

def init_colorama():
    init()

def str2arr(str):
   ret = np.array(str.replace(' ',',').split(','),dtype=int).reshape(-1,2)
   return (ret)

def bold_print(text):
    print (Style.RESET_ALL+Style.BRIGHT+text+Style.RESET_ALL)

def dim_print(text):
    print (Style.RESET_ALL+Style.DIM+text+Style.RESET_ALL)

def success_print(text):
    print (Style.RESET_ALL+Fore.GREEN+text+Style.RESET_ALL)

def fail_print(text):
    print (Style.RESET_ALL+Fore.RED+text+Style.RESET_ALL)

def process_config():
    if not g.args['input'] and not g.args['eventid'] and not g.args['from'] and not g.args['to']:
        fail_print ('Error: You either need to specify an input video, or an event id, or a timeline')
        exit(1)
    if g.args['eventid'] and not ( g.args['username'] and g.args['password'] and g.args['portal'] ):
        fail_print ('Error: If you specify an event ID, you MUST specify username,password and portal')
        exit(1)
    if (g.args['from'] or g.args['to']) and not ( g.args['username'] and g.args['password'] and g.args['portal'] ):
        fail_print ('Error: If you specify a timeline you MUST specify username,password and portal')
        exit(1)    
    if g.args['mask']:
        parr = str2arr(g.args['mask'])
        if g.args['resize']:
            parr =(parr*resize).astype(int)
        g.poly_mask = Polygon(parr)
    
    if g.args['find']:
        g.template = cv2.imread(g.args['find'])
        if g.args['resize']:
            rh, rw, rl = g.template.shape
            g.template = cv2.resize(g.template, (int(rw*resize), int(rh*resize)))
        g.template = cv2.cvtColor(g.template, cv2.COLOR_BGR2GRAY)

    if g.args['threshold']:
        if g.args['threshold'] < g.MIN_ACCURACY:
            # like I said earlier, anything < 0.7 seems to be a 'non match'
            print ('Accuracy too low, setting to {}'.format(g.MIN_ACCURACY))
            g.args['threshold'] = g.MIN_ACCURACY
    else:
        g.args['threshold'] = g.MIN_ACCURACY

    # if either from or to is specified, populate both ranges
    # if not, leave it as None
    if g.args['to'] or g.args['from']:
        if g.args['to']:
            to_time = dateparser.parse(g.args['to'])
        else:
            to_time= datetime.now()
            
        if g.args['from']:
            from_time= dateparser.parse(g.args['from'])
        else:
            from_time = to_time - timedelta(hours = 1)

        g.args['from']= from_time.strftime('%Y-%m-%d %H:%M:%S')
        g.args['to'] = to_time.strftime('%Y-%m-%d %H:%M:%S')
    if (to_time <= from_time):
        fail_print ('ERROR: Time range from:{} to:{} is invalid '.format(from_time, to_time))
        exit(1)

    if g.args['monitors']:
        g.mon_list = [int(item) for item in g.args['monitors'].split(',')]

    
    if not g.args['find'] and not g.args['blend']:
        fail_print('You need to specify either --find or --blend')
        exit(1)



    
