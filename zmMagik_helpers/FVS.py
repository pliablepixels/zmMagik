from imutils.video import FileVideoStream
import cv2

class FileVideoStream(FileVideoStream):
    def get_stream_object(self):
        return self.stream
        
    def get_wh(self):
        w = self.stream.get(3)
        h = self.stream.get(4)
        print ("W={}, H={}".format(w,h))
        print ('TOTAL FRAMES={}'.format(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)))