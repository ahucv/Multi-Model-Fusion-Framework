"""
Object Tracking with multi-models simultaneously
Created by lolimay at 2019.08.20
"""

import os
import multiprocessing
import torch
import os, sys
import torchvision
import torchvision.transforms as transforms
from utils.tracking_with_siamfc import tracking_with_siamfc
import numpy as np

# Use GPU if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tracking(model_index, frame):
    for index, item in enumerate(model_list):
        # load model
        net = torch.load(model)
        net = net.to(device) # Move the model to GPU for calculation
        
        # evaluation mode
        net.eval()

        # load sequence
        img_list, target_position, target_size = load_sequence(p.seq_base_path, p.video)

        # do tracking
        bboxes = tracking_with_siamfc(len(img_list), dtype=np.double)

        for bbox in bboxes:
            best_bbox = find_the_best_bbox(bbox)
            predicted_bboxes.append(best_bbox)

if __name__ == '__main__':
    model_path = "../../models"
    model_list = []
    model_num = None
    predicted_bboxes = [] # tracking result will be stored in this list
    dataset = os.path('../dataset')

    # Add models to model_list
    for model in os.listdir(model_path):
        if (model.endswith('.pth')):
            model_list.append(model)
    model_num = len(model_list)

    for i in range(model_num):
        p = multiprocessing.Process(target=tracking, args=(i)) # tracking in parallel
        p.start()

