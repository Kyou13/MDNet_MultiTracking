# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

sys.path.insert(0,'../modules')
from sample_generator import *
from utils import *

class RegionDataset(data.Dataset):
    def __init__(self, img_dir, img_list, gt_1, gt_2, opts):

        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
        self.gt_1 = gt_1
        self.gt_2 = gt_2

        self.batch_frames = opts['batch_frames'] # 8
        self.batch_pos = opts['batch_pos'] # 32
        self.batch_neg = opts['batch_neg'] # 96
        
        self.overlap_pos = opts['overlap_pos'] # [0.7, 1]
        self.overlap_neg = opts['overlap_neg'] # [0, 0.5]

        self.crop_size = opts['img_size'] # 107
        self.padding = opts['padding'] # 16

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0
        
        image = Image.open(self.img_list[0]).convert('RGB')
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, True)

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer] # image idx をランダムに並び替えたもの 
        # img_listの残りがbactch_frames以下になったら
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer # self.batch_framesずつインクリメント

        regions = np.empty((0,3,self.crop_size,self.crop_size)) # (0,3,107,107)
        labels = np.empty(0)
        for i, (img_path, bbox_1, bbox_2) in enumerate(zip(self.img_list[idx], self.gt_1[idx], self.gt_2[idx])):
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)
            # batch_pos:32 batch_frames:8 //:商
            # 1batchあたりのbbox数計算
            #n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)
            #n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i)
            n_pos = 4
            n_neg = 12
            # overlap_pos:[0.7,1]
            # bbox 座標計算
            # shape(4,4) 
            pos_examples_1 = gen_samples(self.pos_generator, bbox_1, n_pos, overlap_range=self.overlap_pos)
            pos_examples_2 = gen_samples(self.pos_generator, bbox_2, n_pos, overlap_range=self.overlap_pos)
            # shape(12,4)
            neg_examples_1 = gen_samples(self.neg_generator, bbox_1, n_neg, overlap_range=self.overlap_neg)
            neg_examples_2 = gen_samples(self.neg_generator, bbox_2, n_neg, overlap_range=self.overlap_neg)
            # 連結 
            # shape(32*n, 3, 107, 107)
            # bboxの画像crop
            regions = np.concatenate((regions, self.extract_regions(image, pos_examples_1), self.extract_regions(image, neg_examples_1), self.extract_regions(image, pos_examples_2), self.extract_regions(image, neg_examples_2)),axis=0)
            labels = np.concatenate((labels,[0]*len(pos_examples_1),[2]*len(neg_examples_1),[1]*len(pos_examples_2),[2]*len(neg_examples_2)), axis=0)

        regions = torch.from_numpy(regions).float()
        labels = torch.from_numpy(labels).long()
        dataset = torch.utils.data.TensorDataset(regions, labels)
        # return regions, labels
        return dataset
    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)
        
        regions = regions.transpose(0,3,1,2)
        regions = regions.astype('float32') - 128.
        return regions
