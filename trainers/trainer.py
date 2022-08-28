import os
import time

import cv2
import numpy as np
import skimage.metrics
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.utils import save_session, AverageMeter

import pdb

cuda = torch.cuda.is_available()

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, test_loader, args):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.criteria = self.setup_criteria()
        if cuda:
            for name in self.criteria:
                self.criteria[name] = nn.DataParallel(self.criteria[name]).cuda()
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'log'))
        self.step = {
                'train': 0,
                'validate': 0
                }

    def setup_criteria(self):
        criteria = {}
        criteria['flr'] = torch.nn.L1Loss()
        criteria['fhr'] = torch.nn.L1Loss()
        return criteria

    def setup_records(self):
        records = {}
        records['time'] = AverageMeter()
        records['total'] = AverageMeter()
        return records

    def compute_flr_loss(self, pred, batch, stage='train'):
        step = self.step[stage]
        loss_flr = self.criteria['flr'](pred['sharp_frame_lr'],
                                        batch['sharp_frame_lr']).mean()
        self.writer.add_scalar(stage + '/frame_low_resolution', loss_flr, step)
        loss = self.args.lambda_flr * loss_flr
        return loss

    def compute_fhr_loss(self, pred, batch, stage='train'):
        step = self.step[stage]
        loss_fhr = self.criteria['fhr'](pred['sharp_frame_hr'],
                                        batch['sharp_frame_hr']).mean()
        self.writer.add_scalar(stage + '/frame_high_resolution', loss_fhr, step)
        loss = self.args.lambda_fhr * loss_fhr
        return loss

    def compute_losses(self, pred, batch, records, stage='train'):
        loss = 0.
        batch_size = batch['blurry_frame'].shape[0]
        info = []
        if self.args.lambda_flr > 0:
            loss = loss + self.compute_flr_loss(pred, batch, stage)
        if self.args.lambda_fhr > 0:
            loss = loss + self.compute_fhr_loss(pred, batch, stage)
        records['total'].update(loss.detach().cpu().numpy(), batch_size)
        info.append('Total: {:.3f} ({:.3f})'.format(records['total'].val,
                                                    records['total'].avg))
        info = '\t'.join(info)
        step = self.step[stage]
        self.writer.add_scalar(stage + '/total', loss, step)
        self.writer.flush()
        self.step[stage] += 1
        return loss, info

    def train(self, epoch):
        for key in self.model.keys():
            self.model[key].train()
        records = self.setup_records()
        num_iters = min(len(self.train_loader), self.args.iters_per_epoch)
        for i_batch, batch in enumerate(self.train_loader):
            start_time = time.time()
            if i_batch >= self.args.iters_per_epoch:
                break
            pred = self.model['pred'](batch,
                                      predict_lr=self.args.lambda_flr > 0,
                                      predict_hr=self.args.lambda_fhr > 0)
            loss, loss_info = self.compute_losses(pred,
                                                  batch,
                                                  records,
                                                  stage='train')

            self.optimizer['pred'].zero_grad()
            loss.backward()
            self.optimizer['pred'].step()

            # print information during training
            records['time'].update(time.time() - start_time)
            info = 'Epoch: [{}][{}/{}]\t' \
                   'Time: {:.3f} ({:.3f})\t{}'.format(epoch,
                                                      i_batch,
                                                      num_iters,
                                                      records['time'].val,
                                                      records['time'].avg,
                                                      loss_info)
            print(info)

    def test(self, test_hr=True):
        for key in self.model.keys():
            self.model[key].eval()
        metrics = {}
        for metric_name in ['MSE_lr', 'PSNR_lr', 'SSIM_lr',
                            'MSE_hr', 'PSNR_hr', 'SSIM_hr']:
            metrics[metric_name] = AverageMeter()
        with torch.no_grad():
            for i_batch, batch in enumerate(tqdm(self.test_loader)):
                pred = self.model['pred'](batch, predict_hr=test_hr)
                lr_pred = np.clip(pred['sharp_frame_lr'].detach().cpu().numpy(), 0, 1)
                lr_gt = batch['sharp_frame_lr'].detach().cpu().numpy()
                if test_hr:
                    hr_pred = np.clip(pred['sharp_frame_hr'].detach().cpu().numpy(), 0, 1)
                    hr_gt = batch['sharp_frame_hr'].detach().cpu().numpy()
                video_idx = batch['video_idx'].detach().cpu().numpy()
                frame_idx = batch['frame_idx'].detach().cpu().numpy()
                for i_example in range(lr_pred.shape[0]):
                    save_dir = os.path.join(self.args.save_dir,
                                            'output',
                                            '{:03d}'.format(video_idx[i_example]))
                    os.makedirs(os.path.join(save_dir, 'lr'), exist_ok=True)
                    if test_hr:
                        os.makedirs(os.path.join(save_dir, 'hr'), exist_ok=True)
                    for i_time in range(lr_pred.shape[1]):
                        save_name = os.path.join(save_dir,
                                                 'lr',
                                                '{:06d}_{}.png'.format(frame_idx[i_example],
                                                                       i_time))
                        cv2.imwrite(save_name, lr_pred[i_example, i_time] * 255)
                        gt = np.uint8(lr_gt[i_example, i_time] * 255)
                        pred = np.uint8(lr_pred[i_example, i_time] * 255)
                        for metric_name, metric in zip(['MSE_lr', 'PSNR_lr', 'SSIM_lr'],
                                                       [skimage.metrics.normalized_root_mse,
                                                        skimage.metrics.peak_signal_noise_ratio,
                                                        skimage.metrics.structural_similarity]):
                            metrics[metric_name].update(metric(gt, pred))
                        if test_hr:
                            save_name = os.path.join(save_dir,
                                                     'hr',
                                                    '{:06d}_{}.png'.format(frame_idx[i_example],
                                                                           i_time))
                            cv2.imwrite(save_name, hr_pred[i_example, i_time] * 255)
                            gt = np.uint8(hr_gt[i_example, i_time] * 255)
                            pred = np.uint8(hr_pred[i_example, i_time] * 255)
                            for metric_name, metric in zip(['MSE_hr', 'PSNR_hr', 'SSIM_hr'],
                                                           [skimage.metrics.normalized_root_mse,
                                                            skimage.metrics.peak_signal_noise_ratio,
                                                            skimage.metrics.structural_similarity]):
                                metrics[metric_name].update(metric(gt, pred))

        info = 'Low Resolution:\n' \
               'MSE: {:.3f}\tPSNR: {:.3f}\tSSIM: {:.3f}\n' \
               'High Resolution:\n' \
               'MSE: {:.3f}\tPSNR: {:.3f}\tSSIM: {:.3f}'.format(metrics['MSE_lr'].avg,
                                                                metrics['PSNR_lr'].avg,
                                                                metrics['SSIM_lr'].avg,
                                                                metrics['MSE_hr'].avg,
                                                                metrics['PSNR_hr'].avg,
                                                                metrics['SSIM_hr'].avg)
        print('Results:')
        print(info)


    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.args.save_dir, 'checkpoints')
        save_session(self.model, self.optimizer, ckpt_dir, epoch)
