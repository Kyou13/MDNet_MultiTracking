# -*- coding:utf-8 -*-
# groundtruthのtxtデータをpickle化
import os
import numpy as np
import pickle
from collections import OrderedDict

seq_home = '../dataset/'
seqlist_path = 'data/pets-otb.txt'
output_path = 'data/pets-otb.pkl'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()

data = {}
for i,seq in enumerate(seq_list):
    # 画像list化読み込み
    img_list = sorted([p for p in os.listdir(seq_home+seq+'/img') if (os.path.splitext(p)[1] == '.jpg') or (os.path.splitext(p)[1] == '.png')])
    gt_1 = np.loadtxt(seq_home+seq+'/gt_1.txt',delimiter=',')
    gt_2 = np.loadtxt(seq_home+seq+'/gt_2.txt',delimiter=',')

    assert len(img_list) == len(gt_1) , "gt_1, Lengths do not match!! {}".format(seq)
    assert len(img_list) == len(gt_2) , "gt_2, Lengths do not match!! {}".format(seq)
    
    data[seq] = {'images':img_list, 'gt_1':gt_1, 'gt_2':gt_2}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
