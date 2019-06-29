from colorama import init, Fore, Style
from shapely.geometry import Polygon
import cv2
import dateparser
import configargparse
import numpy as np
from datetime import datetime, timedelta
import re


import zmMagik_helpers.globals as g

#https://stackoverflow.com/a/43357954/1361529
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def hist_match(source, template):
#https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(olddtype)

    return interp_t_values[bin_idx].reshape(oldshape)

def init_colorama():
    init()

def secure_string(str):
    return re.sub(r'(((pass)(?:word)?)|(auth)|(token))=([^&/?]*)',r'\1=***',str.lower())    

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
            resize = g.args['resize']
            parr =(parr*resize).astype(int)
        g.raw_poly_mask = parr
        g.poly_mask = Polygon(parr)
    
    if g.args['find']:
        g.template = cv2.imread(g.args['find'])
        if g.args['resize']:
            rh, rw, rl = g.template.shape
            g.template = cv2.resize(g.template, (int(rw*resize), int(rh*resize)))
        g.template = cv2.cvtColor(g.template, cv2.COLOR_BGR2GRAY)


    # if either from or to is specified, populate both ranges
    # if not, leave it as None
    if g.args['to'] or g.args['from']:
        if g.args['to']:
            to_time = dateparser.parse(g.args['to'])
        else:
            to_time= datetime.now()
            
        if g.args['from']:
            from_time = dateparser.parse(g.args['from'])
            print (from_time)
        else:
            from_time = to_time - timedelta(hours = 1)

        g.args['from']= from_time.strftime('%Y-%m-%d %H:%M:%S')
        g.args['to'] = to_time.strftime('%Y-%m-%d %H:%M:%S')
        if (to_time <= from_time):
            fail_print ('ERROR: Time range from:{} to:{} is invalid '.format(from_time, to_time))
            exit(1)

    if g.args['monitors']:
        g.mon_list = [int(item) for item in g.args['monitors'].split(',')]

    if g.args['minblendarea']:
        g.min_blend_area = g.args['minblendarea']

    if not g.args['find'] and not g.args['blend'] and not g.args['annotate']:
        fail_print('You need to specify one of  --find or --blend or --annotate')
        exit(1)


def write_text(frame=None, text=None, x=None,y=None, W=None, H=None, adjust=False):
    (tw, th) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=g.args['fontscale'], thickness=2)[0]
    loc_x1 = x
    loc_y1 = y - th - 4
    loc_x2 = x + tw + 4
    loc_y2 = y

    if adjust:
        if not W or not H:
            fail_print('cannot auto adjust text as W/H  not provided')
        else:
            if loc_x1 + tw > W:
                loc_x1 = max (0, loc_x1 - (loc_x1+tw - W))
            if loc_y1 + th > H:
                loc_y1 = max (0, loc_y1 - (loc_y1+th - H))

    cv2.rectangle(frame, (loc_x1, loc_y1), (loc_x1+tw+4,loc_y1+th+4), (0,0,0), cv2.FILLED)
    cv2.putText(frame, text, (loc_x1+2, loc_y2-2), cv2.FONT_HERSHEY_PLAIN, fontScale=g.args['fontscale'], color=(255,255,255), thickness=1)
    return loc_x1, loc_y1, loc_x1+tw+4,loc_y1+th+4

    
