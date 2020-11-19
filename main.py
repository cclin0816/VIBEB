import time
import shutil
import os
from multiprocessing import Process, Pipe
import tqdm
from lib.HitPoint import HitPoint
from lib.Render import Render, skeleton
from lib.VIBE import VIBE, get_dist
from lib.TrackNet import TrackNet
from util.FrameReader import FR
import numpy as np
import sys
my_stdout = sys.stdout
sys.stdout = open("/dev/null", "w")
import cv2

def Run_TrackNet(pipe, vid_path):
    try:
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
        traj = tn.get_result()
        pipe.send(traj)
    except:
        pipe.send("breakdown")


def Run_HitPoint(pipe, traj):
    try:
        print("Running HitPoint")
        hp, traj = HitPoint(traj)
        pipe.send((hp, traj))
    except:
        pipe.send(("breakdown", 0))


def Run_VIBE(pipe, vid_path):
    try:
        print("Running VIBE")
        vibe_result = VIBE(vid_path)
        pipe.send(vibe_result)
    except:
        pipe.send("breakdown")


def Run_Render(pipe, hp, vibe_result, vid_path, traj):
    try:
        print("Running Render")
        if not os.path.isdir('data/render'):
            os.makedirs('data/render')
        else:
            shutil.rmtree('data/render')
            os.makedirs('data/render')
        new_hp = []
        if get_dist(traj[hp[0]], vibe_result[0]['bboxes'][hp[0]][:2]) > get_dist(traj[hp[0]], vibe_result[1]['bboxes'][hp[0]][:2]):
            k = 1
        else:
            k = 0
        for i in range(len(hp)):
            new_hp.append(((i + k) % 2, hp[i]))
            os.makedirs(f'data/render/{hp[i]}')
        Render(new_hp, vibe_result)
        skeleton(new_hp, vibe_result, vid_path)
        pipe.send("fin")
    except:
        pipe.send("breakdown")


class breakdown(Exception):
    pass


def main():

    try:
        vid_path = sys.argv[1]

        pipe_p, pipe_c = Pipe()

        p = Process(target=Run_TrackNet, args=(pipe_c, vid_path,))
        p.start()
        traj = pipe_p.recv()
        p.join()
        if(traj is "breakdown"):
            raise breakdown

        time.sleep(2)
        p = Process(target=Run_HitPoint, args=(pipe_c, traj,))
        p.start()
        hp, traj = pipe_p.recv()
        p.join()
        if(hp is "breakdown"):
            raise breakdown

        time.sleep(2)
        p = Process(target=Run_VIBE, args=(pipe_c, vid_path,))
        p.start()
        vibe_result = pipe_p.recv()
        p.join()
        if(vibe_result is "breakdown"):
            raise breakdown

        time.sleep(2)
        p = Process(target=Run_Render, args=(
            pipe_c, hp, vibe_result, vid_path, traj))
        p.start()
        render_fin = pipe_p.recv()
        p.join()
        if(render_fin is "breakdown"):
            raise breakdown
        
        print("finish", file=my_stdout)

    except:
        print("breakdown", file=my_stdout)


if __name__ == '__main__':
    main()
