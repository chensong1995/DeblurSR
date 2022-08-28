import math

import torch
import torch.nn as nn

import pdb

class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=10):
        super(UNet, self).__init__()
        self.en_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.en_conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.en_conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.en_conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
                )
        self.res1 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512)
                )
        self.relu1 = nn.ReLU(inplace=True)
        self.res2 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512)
                )
        self.relu2 = nn.ReLU(inplace=True)
        self.de_conv1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=(0, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.de_conv2 = nn.Sequential(
                nn.ConvTranspose2d(256 + 256, 128, kernel_size=5, stride=2, padding=2, output_padding=(0, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.de_conv3 = nn.Sequential(
                nn.ConvTranspose2d(128 + 128, 64, kernel_size=5, stride=2, padding=2, output_padding=(1, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.de_conv4 = nn.Sequential(
                nn.ConvTranspose2d(64 + 64, 32, kernel_size=5, stride=2, padding=2, output_padding=(1, 1), bias=False),
                nn.ReLU(inplace=True)
                )
        self.pred = nn.Conv2d(32 + in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, *args):
        x_in = torch.cat(args, dim=1)
        x1 = self.en_conv1(x_in)
        x2 = self.en_conv2(x1)
        x3 = self.en_conv3(x2)
        x4 = self.en_conv4(x3)
        x5 = self.relu1(self.res1(x4) + x4)
        x6 = self.relu2(self.res2(x5) + x5)
        x7 = self.de_conv1(x6)
        x8 = self.de_conv2(torch.cat([x7, x3], dim=1))
        x9 = self.de_conv3(torch.cat([x8, x2], dim=1))
        x10 = self.de_conv4(torch.cat([x9, x1], dim=1))
        x_out = self.pred(torch.cat([x10, x_in], dim=1))
        return x_out

class ResBlock(nn.Module):
    def __init__(self, channels=512):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        x2 = self.linear2(x1)
        return x + x2

class PredNet(nn.Module):
    def __init__(self,
                 feature_channels=512,
                 hidden_channels=512,
                 event_channels=26,
                 height=180,
                 width=240,
                 predict_count=14,
                 segment_count=20,
                 auto_keypoints=True,
                 kernel=True,
                 normalize=True):
        super(PredNet, self).__init__()
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.event_channels = event_channels
        self.height = height
        self.width = width
        self.predict_count = predict_count
        self.segment_count = segment_count
        self.auto_keypoints = auto_keypoints
        self.kernel = kernel
        self.normalize = normalize
        # prediction networks
        self.unet = UNet(in_channels=1 + event_channels,
                         out_channels=feature_channels)
        self.feature_net = nn.Linear(feature_channels, hidden_channels)
        self.coord_net = nn.Linear(2, hidden_channels)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.resnet = nn.Sequential(
                ResBlock(channels=hidden_channels),
                nn.ReLU(),
                ResBlock(channels=hidden_channels),
                nn.ReLU(),
                ResBlock(channels=hidden_channels),
                nn.ReLU(),
                ResBlock(channels=hidden_channels),
                nn.ReLU(),
                ResBlock(channels=hidden_channels),
                nn.ReLU()
                )
        if kernel:
            assert auto_keypoints
            self.param_net = nn.Linear(hidden_channels, segment_count * (1 + 2 * 9))
        elif auto_keypoints:
            self.param_net = nn.Linear(hidden_channels, segment_count * 3)
        else:
            self.param_net = nn.Linear(hidden_channels, segment_count * 2)
        # coordinate buffers
        x_lr = torch.linspace(-1., 1., width)
        y_lr = torch.linspace(-1., 1., height)
        coords_lr = torch.cartesian_prod(y_lr, x_lr) # [h * w, 2]
        self.register_buffer('coords_lr', coords_lr)
        x_hr = torch.linspace(-1., 1., width * 4)
        y_hr = torch.linspace(-1., 1., height * 4)
        coords_hr = torch.cartesian_prod(y_hr, x_hr) # [4h * 4w, 2]
        self.register_buffer('coords_hr', coords_hr)

    def reconstruct_v1(self, keypoints, timestamps, blurry_frame,
                       slope, intercept, height, width):
        result = torch.zeros(timestamps.shape[0],
                             self.predict_count,
                             height,
                             width).to(timestamps.device)
        # convert timestamps to [bs, T, 1, 1]
        timestamps = timestamps.unsqueeze(dim=-1).unsqueeze(dim=-1)
        for i in range(keypoints.shape[1] - 1):
            start_ = keypoints[:, i].unsqueeze(dim=1)
            end_ = keypoints[:, i + 1].unsqueeze(dim=1)
            slope_ = slope[:, :, :, i].unsqueeze(dim=1)
            intercept_ = intercept[:, :, :, i].unsqueeze(dim=1)
            residual = slope_ * timestamps + intercept_
            validity = (timestamps >= start_) & (timestamps < end_)
            result = result + residual * validity
        # normalization
        if self.normalize:
            offset = torch.mean(result, dim=1, keepdim=True) - blurry_frame
            result = result - offset
        return result

    def reconstruct_v2(self, keypoints, timestamps, blurry_frame,
                       slope, intercept, height, width):
        result = torch.zeros(timestamps.shape[0],
                             self.predict_count,
                             height,
                             width).to(timestamps.device)
        # convert timestamps to [bs, T, 1, 1]
        timestamps = timestamps.unsqueeze(dim=-1).unsqueeze(dim=-1)
        for i in range(keypoints.shape[3] - 1):
            start_ = keypoints[:, :, :, i].unsqueeze(dim=1)
            end_ = keypoints[:, :, :, i + 1].unsqueeze(dim=1)
            slope_ = slope[:, :, :, i].unsqueeze(dim=1)
            intercept_ = intercept[:, :, :, i].unsqueeze(dim=1)
            residual = slope_ * timestamps + intercept_
            validity = (timestamps >= start_) & (timestamps < end_)
            result = result + residual * validity
        # normalization
        if self.normalize:
            offset = torch.mean(result, dim=1, keepdim=True) - blurry_frame
            result = result - offset
        return result

    def get_shifted_frames(self, frame):
        # frame: [bs, C, h, w]
        left = torch.cat([frame[:, :, :, :1], frame[:, :, :, :-1]], dim=-1)
        right = torch.cat([frame[:, :, :, 1:], frame[:, :, :, -1:]], dim=-1)
        up = torch.cat([frame[:, :, :1], frame[:, :, :-1]], dim=-2)
        down = torch.cat([frame[:, :, 1:], frame[:, :, -1:]], dim=-2)
        up_left = torch.cat([left[:, :, :1], left[:, :, :-1]], dim=-2)
        up_right = torch.cat([right[:, :, :1], right[:, :, :-1]], dim=-2)
        down_left = torch.cat([left[:, :, 1:], left[:, :, -1:]], dim=-2)
        down_right = torch.cat([right[:, :, 1:], right[:, :, -1:]], dim=-2)
        shifted = [up_left, up, up_right,
                   left, frame, right,
                   down_left, down, down_right]
        shifted = torch.stack(shifted, dim=-1)
        return shifted

    def reconstruct_v3(self, keypoints, timestamps, blurry_frame,
                       slope, intercept, height, width):
        result = torch.zeros(timestamps.shape[0],
                             self.predict_count,
                             height,
                             width).to(timestamps.device)
        shifted = self.get_shifted_frames(blurry_frame)
        # convert timestamps to [bs, T, 1, 1]
        timestamps = timestamps.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        for i in range(keypoints.shape[3] - 1):
            start_ = keypoints[:, :, :, i].unsqueeze(dim=1).unsqueeze(dim=-1)
            end_ = keypoints[:, :, :, i + 1].unsqueeze(dim=1).unsqueeze(dim=-1)
            slope_ = slope[:, :, :, i].unsqueeze(dim=1)
            intercept_ = intercept[:, :, :, i].unsqueeze(dim=1)
            residual = slope_ * timestamps + intercept_
            validity = (timestamps >= start_) & (timestamps < end_)
            residual = residual * validity
            residual = torch.sum(residual * shifted, dim=-1)
            result = result + residual
        # normalization
        if self.normalize:
            offset = torch.mean(result, dim=1, keepdim=True) - blurry_frame
            result = result - offset
        return result

    def forward(self, batch, predict_lr=True, predict_hr=True):
        # extract image features
        feature = self.unet(batch['blurry_frame'], batch['event_map'])
        # assemble configuration
        config = []
        if predict_lr:
            config.append((feature, self.coords_lr, self.height, self.width,
                           batch['blurry_frame'], 'sharp_frame_lr'))
        if predict_hr and self.auto_keypoints:
            feature_hr = self.upsample(feature)
            blurry_hr = self.upsample(batch['blurry_frame'])
            config.append((feature_hr, self.coords_hr,
                           self.height * 4, self.width * 4,
                           blurry_hr, 'sharp_frame_hr'))
        result = {}
        for feature, coords, height, width, blurry_frame, label in config:
            # [bs, c, h, w] -> [bs, h, w, c] -> [bs * h * w, c]
            feature = feature.permute(0, 2, 3, 1).contiguous()
            feature = feature.reshape(-1, self.feature_channels)

            # copy coordinates for the batch
            coords = torch.stack([coords,] * batch['event_map'].shape[0], dim=0)
            coords = coords.reshape(-1, 2)

            # pass to resnet
            res_in = self.relu(self.feature_net(feature) + \
                               self.coord_net(coords))
            res_out = self.resnet(res_in)

            # predict line segment parameters
            params = self.param_net(res_out)
            params = params.reshape(batch['event_map'].shape[0],
                                    height,
                                    width,
                                    self.segment_count,
                                    -1)

            if self.auto_keypoints:
                weights = torch.sigmoid(params[:, :, :, :, 0])
                keypoints = torch.zeros(batch['event_map'].shape[0],
                                        height,
                                        width,
                                        self.segment_count + 1).to(weights.device)
                for i in range(self.segment_count):
                    keypoints[:, :, :, i + 1] = keypoints[:, :, :, i] + \
                                                weights[:, :, :, i]
                keypoints = keypoints / keypoints[:, :, :, -1:]
                keypoints = keypoints * 2. - 1.
                if self.kernel:
                    slope = torch.tan(torch.tanh(params[:, :, :, :, 1:10]) * math.pi / 2)
                    intercept = params[:, :, :, :, 10:]
                    result[label] = self.reconstruct_v3(keypoints,
                                                        batch['timestamps'],
                                                        blurry_frame,
                                                        slope,
                                                        intercept,
                                                        height,
                                                        width)
                else:
                    slope = torch.tan(torch.tanh(params[:, :, :, :, 1]) * math.pi / 2)
                    intercept = params[:, :, :, :, 2]
                    result[label] = self.reconstruct_v2(keypoints,
                                                        batch['timestamps'],
                                                        blurry_frame,
                                                        slope,
                                                        intercept,
                                                        height,
                                                        width)
            else:
                slope = torch.tan(torch.tanh(params[:, :, :, :, 0]) * math.pi / 2)
                intercept = params[:, :, :, :, 1]
                result[label] = self.reconstruct_v1(batch['keypoints'],
                                                    batch['timestamps'],
                                                    blurry_frame,
                                                    slope,
                                                    intercept,
                                                    height,
                                                    width)
        if predict_hr and not self.auto_keypoints:
            result['sharp_frame_hr'] = self.upsample(result['sharp_frame_lr'])
        return result
