import torch
from multi_person_tracker import MPT

class VIBE():

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def run(self):
        mot = MPT(
            device=self.device,
            output_format='dict'
        )