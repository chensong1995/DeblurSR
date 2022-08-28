import itertools
import os

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

import pdb

class REDSDataset(Dataset):
    def __init__(self,
                 hdf5_name='data/REDS/train_0.hdf5'):
        super(REDSDataset, self).__init__()
        self.data = h5py.File(hdf5_name, 'r')

    def __len__(self):
        return len(self.data['video_idx'])

    def __getitem__(self, idx):
        video_idx = self.data['video_idx'][idx]
        frame_idx = self.data['frame_idx'][idx]
        blurry_frame = self.data['blurry_frame'][idx]
        event_map = self.data['event_map'][idx]
        keypoints = self.data['keypoints'][idx]
        sharp_frame_idx = self.data['sharp_frame_idx'][idx]
        sharp_frame_lr = np.squeeze(self.data['sharp_frame'][sharp_frame_idx], axis=1)
        sharp_frame_hr = np.squeeze(self.data['sharp_frame_hr'][sharp_frame_idx], axis=1)
        timestamps = np.float32(self.data['sharp_frame_ts'][idx])
        item = {
                'video_idx': video_idx, # scalar
                'frame_idx': frame_idx, # scalar
                'blurry_frame': blurry_frame, # (1, 180, 240)
                'event_map': event_map, # (26, 180, 240)
                'keypoints': keypoints, # (10, 180, 240)
                'sharp_frame_lr': sharp_frame_lr, # (14, 180, 240)
                'sharp_frame_hr': sharp_frame_hr, # (14, 720, 960)
                'timestamps': timestamps # (14,)
                }
        return item

if __name__ == '__main__':
    dataset = REDSDataset()
    item = dataset[0]
    pdb.set_trace()
    print(123)
