import torch
from multi_person_tracker import MPT
from multi_person_tracker.data import video_to_images
# import os
# from lib.PoseTracker import *
from lib.models.vibe import VIBE_Demo
from lib.utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    download_ckpt,
)
from tqdm import tqdm
import time
from lib.dataset.inference import Inference
from torch.utils.data import DataLoader
from util.FrameReader import FR
import os
import joblib


class VIBE():

    def __init__(self):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def run(self, vid_path, image_folder):
        fr = FR(vid_path)
        num_frames, img_shape = fr.decode(image_folder)
        orig_width, orig_height = img_shape
        bbox_scale = 1.1

        mot = MPT(device=self.device, output_format='dict')
        tracking_results = mot('data/frame')

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < 25:
                del tracking_results[person_id]
        model = VIBE_Demo(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(self.device)

        pretrained_file = download_ckpt(use_3dpw=False)
        ckpt = torch.load(pretrained_file)
        print(
            f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
        ckpt = ckpt['gen_state_dict']
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        print(f'Loaded pretrained weights from \"{pretrained_file}\"')

        print(f'Running VIBE on each tracklet...')
        vibe_time = time.time()
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

            dataloader = DataLoader(dataset, batch_size=12, num_workers=4)

            with torch.no_grad():
                # batch = []
                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [
                ], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(self.device)

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
        output_path = 'data/'
        joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))
