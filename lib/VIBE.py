import torch

class VIBE():

    def __init__(self):
        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        