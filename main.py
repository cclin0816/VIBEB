import numpy as np
from util.FrameReader import FR
from lib.TrackNet import TrackNet
from lib.VIBE import VIBE
import tqdm
import sys


def main():
    print("Initiaizing...")
    fr = FR('data/video/1_01_00.mp4')
    islast = False
    print("Running TrackNet")
    tn = TrackNet()
    pbar = tqdm.tqdm(total=fr.vid_len, file=sys.stdout)
    while not islast:
        islast, data = fr.read(read_length=12, output_size=(512, 288))
        tn.run(data, 4)
        pbar.update(data.shape[0])
    pbar.close()
    print("Running Smoothing")
    traj = tn.get_result()
    del tn
    print("Running HitPoint")
    print("Running VIBE")
    vibe = VIBE()

if __name__ == '__main__':
    main()
