import argparse
import itertools
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
from tqdm import tqdm

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_time', type=float, default=0.12) # 120ms = 0.12s
    parser.add_argument('--n_kpts', type=int, default=11)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--start_video_idx', type=int, default=0)
    parser.add_argument('--end_video_idx', type=int, default=1)
    parser.add_argument('--out_name', type=str, default='debug.hdf5')
    parser.add_argument('--sharp_dir', type=str, default='resized')
    parser.add_argument('--sharp_hr_dir', type=str, default='resized_hr')
    parser.add_argument('--blurry_dir', type=str, default='corrupted')
    parser.add_argument('--zip_dir', type=str, default='zip')
    args = parser.parse_args()
    return args

def load_events(event_name):
    events = pd.read_csv(event_name,
                         delim_whitespace=True,
                         header=None,
                         names=['t', 'x', 'y', 'pol'],
                         dtype={
                             't': np.float64,
                             'x': np.int16,
                             'y': np.int16,
                             'pol': np.int16
                             },
                         engine='c',
                         skiprows=1)
    return events

def create_voxel(events, num_bins=40, width=32, height=32):
    # events: [t, x, y, p]
    assert(num_bins % 2 == 0) # equal number of positive and negative
    voxel_grid = np.zeros((num_bins, height, width), np.float32)
    if len(events) == 0:
        return voxel_grid
    t_min, t_max = np.min(events[:, 0]), np.max(events[:, 0])
    x = np.int32(events[:, 1])
    y = np.int32(events[:, 2])
    bin_time = (t_max - t_min) / (np.ceil(num_bins / 2))
    for i_bin in range(num_bins):
        t_start = t_min + bin_time * (i_bin // 2)
        t_end = t_start + bin_time
        validity = (events[:, 0] >= t_start) & (events[:, 0] < t_end)    
        if i_bin % 2 == 0:
            validity &= events[:, 3] > 0
        else:
            validity &= events[:, 3] <= 0
        np.add.at(voxel_grid[i_bin], (y[validity], x[validity]), 1)
    return voxel_grid


def extract_keypoints(events, n_kpts=10, width=240, height=180):
    # events: [t, x, y, p]
    # create uniform pivots
    keypoints = np.linspace(-1, 1, n_kpts)[:, None, None]
    interval = keypoints[1] - keypoints[0]
    left, right = keypoints[0], (keypoints[0] + keypoints[1]) / 2
    index, candidate = 0, np.full((height, width), np.nan)
    keypoints = np.tile(keypoints, (1, height, width))
    changes = np.zeros((height, width), np.uint8)
    if len(events) == 0:
        return keypoints
    # normalize timestamps to [-1, 1]
    events[:, 0] = ((events[:, 0] - events[0, 0]) / (events[-1, 0] - events[0, 0]) - 0.5) * 2
    for t, x, y, _ in events:
        x, y = int(x), int(y)
        while t >= right:
            change_mask = ~np.isnan(candidate)
            keypoints[index][change_mask] = candidate[change_mask]
            changes[change_mask] += 1
            candidate = np.full((height, width), np.nan)
            left, right = right, right + interval
            index += 1

        if np.isnan(candidate[y, x]):
            candidate[y, x] = t
        else:
            old_dist = np.abs(candidate[y, x] - keypoints[index, y, x])
            new_dist = np.abs(t - keypoints[index, y, x])
            if old_dist > new_dist:
                candidate[y, x] = t

    change_mask = ~np.isnan(candidate)
    keypoints[index][change_mask] = candidate[change_mask]
    changes[change_mask] += 1
    return keypoints

def check_length(args):
    length = 0
    for i in range(args.start_video_idx, args.end_video_idx):
        blurry_dir = os.path.join(args.blurry_dir,
                                  args.split,
                                  '{:03d}'.format(i))
        length += len(os.listdir(blurry_dir))
    return length

def check_frame_length(args):
    length = 0
    for i in range(args.start_video_idx, args.end_video_idx):
        sharp_dir = os.path.join(args.sharp_dir,
                                 args.split,
                                 '{:03d}'.format(i))
        length += len(os.listdir(sharp_dir))
    return length

def collect_sharp_frame(args, video_idx, writer, write_idx):
    L_meta_name = os.path.join(args.sharp_dir,
                               args.split,
                               '{:03d}'.format(video_idx),
                               'images.csv')
    L_meta = pd.read_csv(L_meta_name,
                         header=None,
                         names=['t', 'name'],
                         engine='c')
    sharp_dict = {}
    for name in L_meta['name']:
        L_name = os.path.join(args.sharp_dir,
                              args.split,
                              '{:03d}'.format(video_idx),
                              name)
        L = np.float64(cv2.imread(L_name, cv2.IMREAD_UNCHANGED)) / 255
        writer['sharp_frame'][write_idx, 0, :, :] = L
        L_hr_name = os.path.join(args.sharp_hr_dir,
                                 args.split,
                                 '{:03d}'.format(video_idx),
                                 name)
        L_hr = np.float64(cv2.imread(L_hr_name, cv2.IMREAD_UNCHANGED)) / 255
        writer['sharp_frame_hr'][write_idx, 0, :, :] = L_hr
        sharp_dict[name] = write_idx
        write_idx += 1
    return write_idx, sharp_dict

def collect_data(args, sharp_dict, video_idx, writer, write_idx):
    event_name = os.path.join(args.zip_dir,
                              args.split,
                              '{:03d}.zip'.format(video_idx))
    events = load_events(event_name)
    B_meta_name = os.path.join(args.zip_dir,
                               args.split,
                               '{:03d}_corrupted_frames.txt'.format(video_idx))
    B_meta = pd.read_csv(B_meta_name,
                         delim_whitespace=True,
                         header=None,
                         names=['idx', 't'],
                         dtype={'idx': np.uint32, 't': np.float64},
                         engine='c')
    L_meta_name = os.path.join(args.sharp_dir,
                               args.split,
                               '{:03d}'.format(video_idx),
                               'images.csv')
    L_meta = pd.read_csv(L_meta_name,
                         header=None,
                         names=['t', 'name'],
                         engine='c')
    for i_B in tqdm(range(len(B_meta))):
        # read blurry frame
        B_name = os.path.join(args.blurry_dir,
                              args.split,
                              '{:03d}'.format(video_idx),
                              '{}.png'.format(i_B))
        B = np.float64(cv2.imread(B_name, cv2.IMREAD_UNCHANGED)) / 255
        height, width = B.shape
        # read sharp timestamps
        exp_start = B_meta['t'][i_B] - args.exp_time / 2
        exp_end = B_meta['t'][i_B] + args.exp_time / 2
        validity = (L_meta['t'] >= exp_start * 1e9) & (L_meta['t'] <= exp_end * 1e9)
        t_L = L_meta['t'][validity].to_numpy() * 1e-9
        timestamps = np.float32(L_meta['t'][validity].to_numpy())
        timestamps = ((timestamps - exp_start * 1e9) / \
                      (exp_end * 1e9 - exp_start * 1e9) - 0.5) * 2
        # find sharp indices
        indices = []
        for name in L_meta['name'][validity].to_list():
            indices.append(sharp_dict[name])
        # create event map
        event_map = np.zeros((26, height, width), dtype=np.float32)
        for i_t in range(len(t_L) - 1):
            event_validity = (events['t'] >= t_L[i_t]) & (events['t'] <= t_L[i_t + 1])
            t = events['t'][event_validity].to_numpy()
            x = events['x'][event_validity].to_numpy()
            y = events['y'][event_validity].to_numpy()
            p = events['pol'][event_validity].to_numpy()
            p[p == 0] = -1
            txyp = np.stack([t, x, y, p], axis=1)
            event_map[i_t*2:i_t*2+2] = create_voxel(txyp,
                                                    num_bins=2,
                                                    width=width,
                                                    height=height)
        # gather all events
        event_validity = (events['t'] >= exp_start) & (events['t'] <= exp_end)
        t = events['t'][event_validity].to_numpy()
        x = events['x'][event_validity].to_numpy()
        y = events['y'][event_validity].to_numpy()
        p = events['pol'][event_validity].to_numpy()
        p[p == 0] = -1
        txyp = np.stack([t, x, y, p], axis=1)
        # extract keypoints
        K = extract_keypoints(txyp,
                              n_kpts=args.n_kpts,
                              width=width,
                              height=height)
        # write to file
        writer['video_idx'][write_idx] = video_idx
        writer['frame_idx'][write_idx] = i_B
        writer['blurry_frame'][write_idx, 0, :, :] = B
        writer['sharp_frame_ts'][write_idx, :] = timestamps
        writer['sharp_frame_idx'][write_idx, :] = indices
        writer['event_map'][write_idx, :, :, :] = event_map
        writer['keypoints'][write_idx, :, :, :] = K
        write_idx += 1
    return write_idx

def main():
    args = parse_args()
    length = check_length(args)
    with h5py.File(args.out_name, 'w', libver='latest') as f:
        frame_length = check_frame_length(args)
        sharp_frame = f.create_dataset('sharp_frame',
                                       (frame_length, 1, 180, 240),
                                       dtype='f',
                                       chunks=(1, 1, 180, 240),
                                       compression='gzip',
                                       compression_opts=9)
        sharp_frame_hr = f.create_dataset('sharp_frame_hr',
                                          (frame_length, 1, 720, 960),
                                          dtype='f',
                                          chunks=(1, 1, 720, 960),
                                          compression='gzip',
                                          compression_opts=9)
        sharp_dict_all = {}
        write_idx = 0
        for i in range(args.start_video_idx, args.end_video_idx):
            write_idx, sharp_dict = collect_sharp_frame(args, i, f, write_idx)
            sharp_dict_all[i] = sharp_dict
        video_idx = f.create_dataset('video_idx', (length,), dtype='i')
        frame_idx = f.create_dataset('frame_idx', (length,), dtype='i')
        blurry_frame = f.create_dataset('blurry_frame',
                                        (length, 1, 180, 240),
                                        dtype='f',
                                        chunks=(1, 1, 180, 240),
                                        compression='gzip',
                                        compression_opts=9)
        frame_ts = f.create_dataset('sharp_frame_ts', (length, 14), dtype='f')
        frame_idx = f.create_dataset('sharp_frame_idx', (length, 14), dtype='i')
        event_map = f.create_dataset('event_map',
                                     (length, 26, 180, 240),
                                     dtype='f',
                                     chunks=(1, 1, 180, 240),
                                     compression='gzip',
                                     compression_opts=9)
        keypoints = f.create_dataset('keypoints',
                                     (length, args.n_kpts, 180, 240),
                                     dtype='f',
                                     chunks=(1, 1, 180, 240),
                                     compression='gzip',
                                     compression_opts=9)
        write_idx = 0
        for i in tqdm(range(args.start_video_idx, args.end_video_idx)):
            write_idx = collect_data(args, sharp_dict_all[i], i, f, write_idx)

if __name__ == '__main__':
    main()
