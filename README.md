What
----
zmMagik will be a list of growing foo-magic things you can do with video images that ZM stores. _Probably..._

Features
---------
As of today, it lets you:
* Find an image fragment inside multiple events. For example, someone stole your amazon package. Crop a picture of an event with that package and then ask zmMagik to search for events where this package went missing. Whoosh!

* Create a blend of multiple events to quickly see how the day went. Swoosh!

Why
----
* I came home one day to see my trash can cover went missing. I thought it would be fun to write a tool that could search through my events to let me know when it went missing. Yep, it started with trash talking

* Andy posted an example of how other vendors blend multiple videos to give a common view quickly. I thought it would be fun to try

Limitations
------------
* Only works with video mp4 files. Did not bother adding support for JPEG store
* Very, very, very, pre-alpha
* Multi-server most likely won't work

Installation
------------

```bash
# needs python3, so you may need to use pip3 if you have 2.x as well
git clone https://github.com/pliablepixels/zmMagik
cd zmMagik
pip install -r requirements.txt
```

If you are using yolo extraction, you also need these files and make sure your config variables point to them
```
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```


Examples
---------

General note: do a `python ./zmMagik -h` to see all options. Remember, you can stuff in regularly used options in a config file and override on CLI as you need. Much easier.

* Create a blended event for monitor 11 in specified time range. config.ini has your ZM usename/password/portal etc so you don't need to type it in every time

```bash
python ./magik.py --monitors=11 --from "yesterday, 7am" --to "today, 10am" --blend -c config.ini
```

* Search when an object (image) went missing:

```bash
python ./magik.py --monitors=7 --from "today, 7am" --to "today, 7pm" --find "amazonpackage.jpg" -c config.ini
```

Note that `amazonpackage.jpg` needs to be the same dimensions/orientation as in the events it will look into. Best way to do this is to load up ZM, find the package, crop it and use it.


FAQ
-----

* Using "background_extraction" mode isn't that great
  * Yeah, its 'pre,pre,pre alpha'
  * Foreground and background are measured based on what is moving (foreground) and what is not (background)
  * Use masks to restrict area
  * Use `--display` to see what is going on, look at frame masks and output
  * Try changing the learning rate of the background extractor
  * See if using a different Background extractor for `fgbg` in `globals.py` helps you (read [this](https://docs.opencv.org/3.3.0/d2/d55/group__bgsegm.html#gae561c9701970d0e6b35ec12bae149814))
  * Fiddle with kernel_clean and kernel_fill in `globals.py`
* Using "Yolo" extraction mode is great, but it overlays complete rectangles
  * Yeah, unlike "background_extraction" yolo doesn't report a mask of the object shape, only a bounding box
  * I'll add masked R-CNN too, you can try that (will be slower than Yolo)
  * Maybe you can suggest a smarter way to overlay the rectangle using some fancy operators that will act like its blending?

* `find` doesn't find my image
  * Congratulations, maybe no one stole your amazon package
  * Make sure image you are looking for is not rotated/resized/etc. needs to be original dimensions

