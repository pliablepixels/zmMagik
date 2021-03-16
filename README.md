# NOT MAINTAINED 
I've run out of time to maintain this software. It is also quite dated and doesn't use pyzm which has many powerful features that this utility will benefit from.


## What
zmMagik will be a list of growing foo-magic things you can do with video images that ZM stores. _probably..._

## Why

* I came home one day to see my trash can cover went missing. I thought it would be fun to write a tool that could search through my events to let me know when it went missing. Yep, it started with trash talking

* Andy posted an example of how other vendors blend multiple videos to give a common view quickly. I thought it would be fun to try

* One thing leads to another and I keep doing new things to learn new things..


## Features

As of today, it lets you:

* **Blend**  multiple events to quickly see how the day went. Imagine compressing 24 hours of video into 1 minute with object overlays. Gadzooks!

<sub><sup>this video is blended from 2 days worth of video. Generated using `python3 ./magik.py -c config.ini  --monitors 11 --blend --display --download=False --from "2 days ago"`</sup></sub>

<img src='https://github.com/pliablepixels/zmMagik/blob/master/sample/1.gif' width=400px />

* **Annotate** recorded ZoneMinder videos. Holy Batman!

<sub><sup>generated using `python3 ./magik.py -c config.ini --eventid 44063 --dumpjson --annotate --display --download=False --onlyrelevant=False --skipframes=1`</sup></sub>

<img src='https://github.com/pliablepixels/zmMagik/blob/master/sample/3.gif' width=400px>

* **Find** an image fragment inside multiple events. For example, someone stole your amazon package. Crop a picture of an event with that package and then ask zmMagik to search for events where this package went missing. Great Krypton!

<sub><sup>generated using `python3 ./magik.py -c config.ini --find trash.jpg --dumpjson  --display --download=False --from "8am" --to "3pm" --monitors 11`</sup></sub>

<img src='https://github.com/pliablepixels/zmMagik/blob/master/sample/2.jpg' width=400px />


## Limitations

* Only works with video mp4 files. Did not bother adding support for JPEG store
* Very Beta. Also, if you don't have a GPU, make sure you play with the flags to optimize skipframes, detection mode, resize
* Multi-server most likely won't work

## Installation


```bash
# needs python3, so you may need to use pip3 if you have 2.x as well
git clone https://github.com/pliablepixels/zmMagik
cd zmMagik
# you may need to do sudo -H pip3 instead for below, if you get permission errors
pip3 install -r requirements.txt
```

Note that this package also needs OpenCV which is not installed by the above step by default. This is because you may have a GPU and may want to use GPU support. If not, pip is fine. See [this page](https://zmeventnotification.readthedocs.io/en/latest/guides/hooks.html#opencv-install) on how to install OpenCV

If you are using yolo extraction, you also need these files and make sure your config variables point to them
```
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```


## Examples


General note: do a `python3 ./zmMagik -h` to see all options. Remember, you can stuff in regularly used options in a config file and override on CLI as you need. Much easier.

* Create a blended event for monitor 11 in specified time range. config.ini has your ZM usename/password/portal etc so you don't need to type it in every time

```bash
python3 ./magik.py --monitors=11 --from "yesterday, 7am" --to "today, 10am" --blend -c config.ini
```

* Search when an object (image) went missing:

```bash
python3 ./magik.py --monitors=7 --present=False --from "today, 7am" --to "today, 7pm" --find "amazonpackage.jpg" -c config.ini
```

Note that `amazonpackage.jpg` needs to be the same dimensions/orientation as in the events it will look into. Best way to do this is to load up ZM, find the package, crop it and use it.


## FAQ

### How do I use GPU acceleration? 
See GPU section below

### What is "mixed" background extraction?
This is the default mode. It uses the very fast openCV background subtraction to detect motion, and then uses YOLO to refine the search to see if it really is an object worth marking. Use this mode by default, unless you need more speed, in which case, use "backround_extraction"

### Using "background_extraction" mode isn't that great...
Yes, that's why you should use "mixed"
Some tips:
  * Use masks to restrict area
  * Use `--display` to see what is going on, look at frame masks and output
  * Try changing the learning rate of the background extractor
  * See if using a different Background extractor for `fgbg` in `globals.py` helps you (read [this](https://docs.opencv.org/3.3.0/d2/d55/group__bgsegm.html#gae561c9701970d0e6b35ec12bae149814))
  * Fiddle with kernel_clean and kernel_fill in `globals.py`

### Using "Yolo" or "mixed" extraction mode is great, but it overlays complete rectangles
Yes, unlike "background_extraction" yolo doesn't report a mask of the object shape, only a bounding box/ I'll eventually add masked R-CNN too, you can try that (will be slower than Yolo)
Maybe you can suggest a smarter way to overlay the rectangle using some fancy operators that will act like its blending?

### `find` doesn't find my image
Congratulations, maybe no one stole your amazon package
  * Make sure image you are looking for is not rotated/resized/etc. needs to be original dimensions

## GPU FAQ

### What is done in the GPU?
Only the DNN object detection part (Yolo). magik uses various image functions like background extraction, merging, resizes etc. that are _not_ done in the GPU. As of today, the OpenCV python CUDA wrapper documentation is dismal and unlike what many people think, where its just a `.cuda` difference in API calls, the flow/parameters also differ. See [this](https://jamesbowley.co.uk/accelerating-opencv-with-cuda-streams-in-python/) for example. I may eventually  get to wrapping all the other non DNN parts into their CUDA equivalent functions, but right now, I'm disinclined to do it. I'll be happy to accept PRs.

### How do I get GPU working?
As of Feb 2020, OpenCV 4.2 is released, which supports CUDA for DNN.
You have two options:

- (RECOMMENDED) Compile and install OpenCV 4.2+ and enable CUDA support. See [this](https://zmeventnotification.readthedocs.io/en/latest/guides/hooks.html#opencv-install) guide on how to do that.

- (LEGACY) Or, compile darknet directly with GPU support. If you go this route, I'd suggest you build the [darknet fork maintained by AlexyAB](https://github.com/AlexeyAB/darknet) as it is better maintained.
  - If you need help compiling darknet for GPU and CUDA 10.x, see [simpleYolo](https://github.com/pliablepixels/simpleYolo). Do NOT use darknet lib directly for a CPU compiled library. It is terribly slow (in my tests, OpenCV was around 50x faster)
  - Only builds of darknet from 2019-10-25 or later should be used with zmMagik due to a change in the darknet data structures

You'd only want to compile darknet directly if your GPU/CUDA version is not compatible with OpenCV 4.2. For all other cases, go with OpenCV 4.2 (don't install from a pip package, GPU is not enabled)


Simply put:
* Either compile OpenCV 4.2+ from source correctly or go the direct darknet route as described before
* Make sure it is actually using GPU
* then set `gpu=True` and either specify `use_opencv_dnn_cua` to `True` or set `darknet_lib=<path/to/filename of gpu accelerated so>`

### How much GPU memory do I need?
The YoloV3 model config I use takes up 1.6GB of GPU memory. Note that I use a reduced footprint yolo config. I have 4GB of GPU memory, so the default `yolov3.cfg` did not work and ate up all my memory. This is my modified `yolov3.cfg` section to make it work:

```
[net]
batch=1
subdivisions=1
width=416
height=416
<and then all the stuff that follows>
```

### How much speed boost can I expect with GPU?

Here is a practical comparison. I ran a blend operation on my driveway camera (modect) for a full day's worth of alarmed events. I used 'mixed' mode, which first used openCV background subtraction and then YOLO if the first mode found anything. This was to be fair to the CPU stats when compared. It grabbed a total of 27 video events:
  ```
  python3 ./magik.py --blend --from "1 day ago"  --monitors 8 -c ./config.ini --gpu=True --alarmonly=True --skipframes=1

  Total time: 250.72s
  ```

  I then ran it without GPU: (Note that I have `libopenblas-dev liblapack-dev libblas-dev` configured with OpenCV to improve CPU performance a lot)
  ```
  python3 ./magik.py --blend --from "1 day ago"  --monitors 8 -c ./config.ini --gpu=False --alarmonly=True --skipframes=1

  Total time: 1234.77s
  ```

  **Thats a 5x improvement**

  * On my 1050 Ti, YoloV3 inferences drops to 120ms or less, compared to 2-3 seconds on GPU
  * That being said, blending/annotating involves:
    * reading frames (A)
    * processing frames (B)
    * writing frames (C)
  * GPU affects point B. If you are reading very large events, A & C will still take its own time. You likely won't see a big improvement there. If there are many objects (B), then obviously, GPU performance improvements will have a huge impact. To make A & C faster:
    * use `resize`
    * use `skipframes`
    * use `alarmonly=True`
    * If you use mocord in ZM, then starting ZM 1.34, set `EVENT_CLOSE_MODE` to "alarm". That will create a new event when an alarm occurs, and close it when the alarm closes. That will help you speed things up a lot 
    * All that being said, I'm using a threaded opencv pipeline to read frames which does improve read performance compared to before (credit to imutils)
