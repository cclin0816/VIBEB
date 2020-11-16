import numpy as np
from util.FrameReader import FR
from lib.TrackNet import TrackNet
from lib.VIBE import VIBE
from lib.Render import Render
from lib.HitPoint import HitPoint
import tqdm
import sys
from multiprocessing import Process, Pipe

def Run_TrackNet(pipe, vid_path):
    print("Running TrackNet")
    fr = FR(vid_path)
    islast = False
    tn = TrackNet()
    pbar = tqdm.tqdm(total=fr.vid_len, file=sys.stdout)
    while not islast:
        islast, data = fr.read(read_length=12, output_size=(512, 288))
        tn.run(data, 4)
        pbar.update(data.shape[0])
    pbar.close()
    print("Running Smoothing")
    traj = tn.get_result()
    pipe.send(traj)

def Run_HitPoint(pipe, traj):
    print("Running HitPoint")
    hp = HitPoint(traj)
    pipe.send(hp)

def Run_VIBE(pipe, vid_path):
    print("Running VIBE")
    vibe_result = VIBE(vid_path)
    pipe.send(vibe_result)

def Run_Render(pipe, hp, vibe_result):
    print("Running Render")
    Render(hp, vibe_result)


def main():
    print("Initiaizing...")

    vid_path = 'data/video/1_01_00.mp4'

    pipe_p, pipe_c = Pipe()

    p = Process(target=Run_TrackNet, args=(pipe_c, vid_path,))
    p.start()
    traj = pipe_p.recv()
    p.join()

    p = Process(target=Run_HitPoint, args=(pipe_c, traj,))
    p.start()
    hp = pipe_p.recv()
    p.join()

    p = Process(target=Run_VIBE, args=(pipe_c, vid_path,))
    p.start()
    vibe_result = pipe_p.recv()
    p.join()

    p = Process(target=Run_Render, args=(pipe_c, hp, vibe_result,))
    p.start()
    # vibe_result = pipe_p.recv()
    p.join()



if __name__ == '__main__':
    main()
