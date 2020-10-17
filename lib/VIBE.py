import torch
from multi_person_tracker import MPT
from multi_person_tracker.data import video_to_images

class VIBE():

    def __init__(self):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def run(self):
        image_folder, num_frames, img_shape = video_to_images('data/video/1_01_00.mp4', 'out', return_info=True)
        orig_height, orig_width = img_shape[:2]
        mot = MPT(device=self.device)
        tracking_results = mot(image_folder, output_file='sample_output.mp4')
        print(tracking_results)