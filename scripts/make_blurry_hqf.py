import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_name', type=str, default='clean_hdf5/bike_bay_hdr.h5')
    parser.add_argument('--out_name', type=str, default='blurry_hdf5/bike_bay_hdr.hdf5')
    parser.add_argument('--n_frames', type=int, default=14)
    parser.add_argument('--n_bins', type=int, default=26)
    args = parser.parse_args()
    return args

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

def collect_data(args, write_idx, f_in, f_out):
    start_idx = write_idx * args.n_frames
    end_idx = (write_idx + 1) * args.n_frames
    sharp_frame = []
    for idx in range(start_idx, end_idx):
        sharp_frame.append(np.float32(f_in['images']['image{:09d}'.format(idx)][:]))
    sharp_frame = np.array(sharp_frame) / 255
    blurry_frame = np.mean(sharp_frame, axis=0)
    t_start = f_in['images']['image{:09d}'.format(start_idx)].attrs['timestamp']
    t_end = f_in['images']['image{:09d}'.format(end_idx - 1)].attrs['timestamp']
    validity = (f_in['events']['ts'] >= t_start) & (f_in['events']['ts'] <= t_end)
    t = f_in['events']['ts'][validity]
    x = f_in['events']['xs'][validity]
    y = f_in['events']['ys'][validity]
    p = np.int32(f_in['events']['ps'][validity])
    p[p == 0] = -1
    txyp = np.stack([t, x, y, p], axis=1)
    event_map = create_voxel(txyp,
                             num_bins=args.n_bins,
                             width=blurry_frame.shape[1],
                             height=blurry_frame.shape[0])
    keypoints = extract_keypoints(txyp,
                                  n_kpts=11,
                                  width=blurry_frame.shape[1],
                                  height=blurry_frame.shape[0])
    f_out['blurry_frame'][write_idx, 0, :, :] = blurry_frame
    f_out['sharp_frame'][write_idx, :, :, :] = sharp_frame
    f_out['event_map'][write_idx, :, :, :] = event_map
    f_out['keypoints'][write_idx, :, :, :] = keypoints

def main():
    args = parse_args()
    with h5py.File(args.in_name, 'r') as f_in:
        length = len(f_in['images']) // args.n_frames
        os.makedirs(os.path.dirname(args.out_name), exist_ok=True)
        with h5py.File(args.out_name, 'w', libver='latest') as f_out:
            blurry_frame = f_out.create_dataset('blurry_frame',
                                                (length, 1, 180, 240),
                                                dtype='f',
                                                chunks=(1, 1, 180, 240),
                                                compression='gzip',
                                                compression_opts=9)
            sharp_frame = f_out.create_dataset('sharp_frame',
                                               (length, args.n_frames, 180, 240),
                                               dtype='f',
                                               chunks=(1, 1, 180, 240),
                                               compression='gzip',
                                               compression_opts=9)
            event_map = f_out.create_dataset('event_map',
                                             (length, args.n_bins, 180, 240),
                                             dtype='f',
                                             chunks=(1, 1, 180, 240),
                                             compression='gzip',
                                             compression_opts=9)
            keypoints11 = f_out.create_dataset('keypoints',
                                               (length, 11, 180, 240),
                                               dtype='f',
                                               chunks=(1, 1, 180, 240),
                                               compression='gzip',
                                               compression_opts=9)
            for i in tqdm(range(length)):
                collect_data(args, i, f_in, f_out)

if __name__ == '__main__':
    main()
