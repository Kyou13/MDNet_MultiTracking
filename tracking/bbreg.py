# -*- coding:utf-8 -*-
import sys
from sklearn.linear_model import Ridge
import numpy as np

from utils import *

class BBRegressor():
    def __init__(self, img_size, alpha=1000, overlap=[0.6, 1], scale=[1, 2]):
        self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.model = Ridge(alpha=self.alpha)

    def train(self, X, bbox, gt):
        X = X.cpu().numpy()
        bbox = np.copy(bbox)
        gt = np.copy(gt)
        
        if gt.ndim==1:
            # 2次元化
            gt = gt[None,:]

        r = overlap_ratio(bbox, gt)
        s = np.prod(bbox[:,2:], axis=1) / np.prod(gt[0,2:])
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])

        # overlap , scale_range満たしたやつとりだす
        X = X[idx]
        bbox = bbox[idx]

        Y = self.get_examples(bbox, gt)
        
        # リッジ回帰学習
        self.model.fit(X, Y)

    def predict(self, X, bbox):
        X = X.cpu().numpy()
        bbox_ = np.copy(bbox)

        # 5 * 4  (-1 ~ 1)
        Y = self.model.predict(X)
    
        # left, top にw/2,h/2の値をプラス
        # 中心座標
        bbox_[:,:2] = bbox_[:,:2] + bbox_[:,2:]/2
        # Ridgeの結果とw,hの積とleft,上の結果を足す
        # = 差分を求める
        bbox_[:,:2] = Y[:,:2] * bbox_[:,2:] + bbox_[:,:2]
        # 対数表示なのでexp
        bbox_[:,2:] = np.exp(Y[:,2:]) * bbox_[:,2:]
        # left,topに戻す
        bbox_[:,:2] = bbox_[:,:2] - bbox_[:,2:]/2
        
        r = overlap_ratio(bbox, bbox_)
        s = np.prod(bbox[:,2:], axis=1) / np.prod(bbox_[:,2:], axis=1)
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])
        # 各要素のTrue, False 判定
        idx = np.logical_not(idx)
        bbox_[idx] = bbox[idx]
 
        bbox_[:,:2] = np.maximum(bbox_[:,:2], 0)
        bbox_[:,2:] = np.minimum(bbox_[:,2:], self.img_size - bbox[:,:2])

        return bbox_
    
    def get_examples(self, bbox, gt):
        # left, top にw/2,h/2の値をプラス
        #   = 中心座標
        bbox[:,:2] = bbox[:,:2] + bbox[:,2:]/2
        gt[:,:2] = gt[:,:2] + gt[:,2:]/2

        # 小さい方が良い
        dst_xy = (gt[:,:2] - bbox[:,:2]) / bbox[:,2:]
        # w,hが近似してるほど0
        dst_wh = np.log(gt[:,2:] / bbox[:,2:])

        Y = np.concatenate((dst_xy, dst_wh), axis=1)
        return Y

