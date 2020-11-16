import numpy as np
from util.FrameReader import FR
from torch.utils.data import DataLoader
from lib.dataset.inference import Inference
from tqdm import tqdm
from lib.utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    download_ckpt,
)
from lib.models.vibe import VIBE_Demo
from multi_person_tracker.data import video_to_images
from multi_person_tracker import MPT
import torch
import math


def get_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)

def get_speed(a, b):
    dx = a[0] + b[0]
    dy = a[1] + b[1]
    return math.sqrt(dx * dx + dy * dy)

def filter_player(tracking_results, num_frames):
    stick = []
    for key in tracking_results:
        first = tracking_results[key]['frames'][0]
        last = tracking_results[key]['frames'][-1]
        f = tracking_results[key]['bbox'][0]
        l = tracking_results[key]['bbox'][-1]
        fcenter = (f[0] + f[2] / 2, f[1] + f[3] / 2)
        lcenter = (l[0] + l[2] / 2, l[1] + l[3] / 2)
        if(len(tracking_results[key]['bbox']) > 1):
            f2 = tracking_results[key]['bbox'][1]
            l2 = tracking_results[key]['bbox'][-2]
            f2center = (f2[0] + f2[2] / 2, f2[1] + f2[3] / 2)
            l2center = (l2[0] + l2[2] / 2, l2[1] + l2[3] / 2)
            fspeed = (f2center[0] - fcenter[0], f2center[1] - fcenter[1])
            lspeed = (lcenter[0] - l2center[0], lcenter[1] - l2center[1])
        else:
            fspeed = (0, 0)
            lspeed = (0, 0)
        stick.append({'first': {'frame': first, 'center': fcenter, 'speed': fspeed}, 'last': {
                     'frame': last, 'center': lcenter, 'speed': lspeed}, 'id': key})
    stick_record = {}
    i = 0
    while i < len(stick):
        stick_record[stick[i]['id']] = [stick[i]['id']]
        j = i + 1
        while j < len(stick):
            if((stick[j]['first']['frame'] > stick[i]['last']['frame']) and
                stick[j]['first']['frame'] - stick[i]['last']['frame'] < 20 and
                    get_dist(stick[j]['first']['center'], stick[i]['last']['center']) < 70 + \
                         get_speed(stick[j]['first']['speed'], stick[i]['last']['speed']) * 3):
                stick_record[stick[i]['id']].append(stick[j]['id'])
                stick[i]['last']['frame'] = stick[j]['last']['frame']
                stick[i]['last']['center'] = stick[j]['last']['center']
                stick[i]['last']['speed'] = stick[j]['last']['speed']
                stick.pop(j)
            else:
                j += 1
        i += 1
    i = 0
    while i < len(stick):
        if(stick[i]['last']['frame'] - stick[i]['first']['frame'] < num_frames / 1.5):
            stick.pop(i)
        else:
            i += 1
    
    my_track = {}
    travel_distance = []
    for st in stick:
        bb = np.zeros((num_frames, 4), dtype=float)
        feeder = stick_record[st['id']][0]
        id = 1
        dist_sum = 0
        prev_pos = (tracking_results[feeder]['bbox'][0][0], tracking_results[feeder]['bbox'][0][1])
        feeder_i = 0
        prev_vallid = 0
        for i in range(num_frames):
            if i < st['first']['frame']:
                for k in range(4):
                    bb[i][k] = tracking_results[feeder]['bbox'][0][k]
            elif i > st['last']['frame']:
                for k in range(4):
                    bb[i][k] = tracking_results[feeder]['bbox'][-1][k]
            elif i == tracking_results[feeder]['frames'][feeder_i]:
                for k in range(4):
                    bb[i][k] = tracking_results[feeder]['bbox'][feeder_i][k]
                prev_vallid = i
                feeder_i += 1
            else: 
                d = i - prev_vallid
                e = tracking_results[feeder]['frames'][feeder_i] - prev_vallid
                for k in range(4):
                    bb[i][k] = bb[prev_vallid][k] + (tracking_results[feeder]['bbox'][feeder_i][k] - bb[prev_vallid][k]) * d / e
            if feeder_i == len(tracking_results[feeder]['frames']) and id < len(stick_record[st['id']]):
                feeder_i = 0
                feeder = stick_record[st['id']][id]
                id += 1
            dist_sum += get_dist(prev_pos, (bb[i][0], bb[i][1]))
            prev_pos = (bb[i][0], bb[i][1])
        my_track[st['id']] = {'bbox': bb, 'frames':np.arange(num_frames, dtype=int)}
        travel_distance.append((dist_sum, st['id']))
    
    travel_distance = sorted(travel_distance, key=lambda x : x[0])
    
    new_track = {}
    new_track[0] = my_track[travel_distance[-1][1]]
    new_track[1] = my_track[travel_distance[-2][1]]

    return new_track

def VIBE(vid_path, image_folder='data/frame/'):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    fr = FR(vid_path)
    num_frames, orig_width, orig_height = fr.decode(image_folder)
    bbox_scale = 1.1

    mot = MPT(device=device, detection_threshold=0.5, output_format='dict')
    tracking_results = mot('data/frame')  # sorted with first frame

    tracking_results = filter_player(tracking_results, num_frames)
    
    # fr.rewind()
    # out = cv2.VideoWriter('mytracker.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (orig_width, orig_height))
    # for i in range(num_frames):
    #     _, img = fr.read_frame()
    #     for id in tracking_results:
    #         d = tracking_results[id]['bbox'][i].astype(np.int32)
    #         cv2.rectangle(
    #             img, (d[0] - d[2] // 2, d[1] - d[3] // 2), (d[0] + d[2] // 2, d[1] + d[3] // 2),
    #             color=(200, 120, 120), thickness=int(round(img.shape[0] / 256))
    #         )
    #     out.write(img)
    # out.release()


    model = VIBE_Demo(
        seqlen=num_frames,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(
        f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    print(f'Running VIBE on each tracklet...')
    # vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        bboxes = tracking_results[person_id]['bbox']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder='data/frame',
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=12)

        with torch.no_grad():
            batch = []
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [
            ], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(
                    batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(
                    batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:, :, 3:75].reshape(
                    batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :, 75:].reshape(
                    batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(
                    batch_size * seqlen, -1, 3))

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        # if args.smooth:
        #     min_cutoff = args.smooth_min_cutoff # 0.004
        #     beta = args.smooth_beta # 1.5
        #     print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
        #     pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
        #                                                     min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    del model
    # output_path = 'data/'
    # joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))
    return vibe_results