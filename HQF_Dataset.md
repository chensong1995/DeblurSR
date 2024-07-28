# Guide to Using the HQF Dataset

## Introduction
This document explains how to use the [HQF](https://timostoff.github.io/20ecnn) dataset. We apply temporal averaging to synthesize blurry images from the original sharp frames. The results will be stored in a few .hdf5 files.

## Download Links
The conversion results are uploaded to [Google Drive](https://drive.google.com/file/d/1-6dJtg3HbWbOWeHVLHFkv4vinmX33Tm-/view?usp=sharing).

The rest of the guide can be skipped entirely if you choose download these .hdf5 files and put them in `data/HQF/`.

## Step 1: Obtain the Original HQF Dataset
From this [link](https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_), download the 14 .bag files. Create a directory called `data/HQF/bag` and move the files here. The final directory structure should look like this:
```
<project root>
  |-- data
  |     |-- HQF
  |     |     |-- bag
  |     |     |     |-- bike_bay_hdr.bag
  |     |     |     |-- boxes.bag
  |     |     |     |-- <many other .bag files>
  |     |     |     |-- still_life.bag
  |-- <other files>
```

## Step 2: Conversion to Clean HDF5
The .bag format is hard to process. We will first use the official tools to convert the .bag files into the HDF5 format. Please follow the instructions under "Conversion to HDF5" in [this document](https://github.com/TimoStoff/event_cnn_minimal). The commands we need is:
```
python events_contrast_maximization/tools/rosbag_to_h5.py <project_root>/data/HQF/bag --output_dir <project_root>/data/HQF/clean_hdf5 --event_topic /dvs/events --image_topic /dvs/image_raw --height 180 --width 240
```
This should create a directory called `data/HQF/clean_hdf5` containing 14 .h5 files.

## Step 3: Conversion to Blurry HDF5
We will now apply temporal average to synthesize blurry frames. Move `scripts/make_blurry_hqf.py` and `scripts/make_blurry_hqf.sh` to `data/HQF`. From there, run `bash make_blurry_hqf.sh`. This should create a directory called `data/HQF/blurry_hdf5` containing 14 .hdf5 files.
