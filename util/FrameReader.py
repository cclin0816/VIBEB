import cv2
import numpy as np
import sys


class FR:

    def __init__(self, vid_path=''):
        self.__vid = cv2.VideoCapture()
        if not self.__vid.open(vid_path):
            sys.exit('Error Opening File')
        self.vid_len = int(self.__vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # return tuple( is last (bool), frame data (np array of fc * 3 * w * h) )
    def read(self, frame_number=-1, read_length=1, output_size=()):
        if frame_number != -1:
            if frame_number < 0 or frame_number >= self.vid_len:
                sys.exit(f'Error Frame Number {frame_number}')
            self.__vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if read_length == -1:
            read_length = self.vid_len
        elif read_length < 1:
            sys.exit(f'Error Read Length {read_length}')

        frame_stack = []
        for _ in range(read_length):
            sucess, frame = self.__vid.read()
            if not sucess:
                if not self.__end_read():
                    sys.exit(
                        f'Error Reading Frame {self.__vid.get(cv2.CAP_PROP_POS_FRAMES)} / {self.vid_len}')
                break
            if output_size != ():
                if not type(output_size) is tuple or len(output_size) > 2 or output_size[0] < 0 or output_size[1] < 0:
                    sys.exit(f'Error Output Size {output_size}')
                frame = cv2.resize(frame, output_size,
                                   interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, -1, 0)
            frame_stack.append(frame)

        return self.__end_read(), np.array(frame_stack)

    def rewind(self):
        self.__vid.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

    def __end_read(self):
        return self.__vid.get(cv2.CAP_PROP_POS_FRAMES) == self.vid_len

    def __del__(self):
        if self.__vid.isOpened():
            self.__vid.release()
