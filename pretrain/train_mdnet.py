# -*- coding: utf-8 -*-
import os
import sys
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_prov import *
from model import *
from options import *

img_home = '../dataset/'
data_path = 'data/pets-otb.pkl'

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    
    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    # 動画数
    K = len(data)
    # [Non,...]
    dataset = [None]*K

    for k, (seqname, seq) in enumerate(data.iteritems()):
        img_list = seq['images']
        gt_1 = seq['gt_1']
        gt_2 = seq['gt_2']
        img_dir = os.path.join(img_home, seqname,"img")
        # class宣言
        # data_prov
        dataset[k] = RegionDataset(img_dir, img_list, gt_1, gt_2, opts)

    ## Init model ##
    ## vggでプレトレーニング
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
        
    ## Init criterion and optimizer ##
    criterion = nn.CrossEntropyLoss() 
    evaluator = Precision() # 精度
    optimizer = set_optimizer(model, opts['lr'])

    best_prec = 0.
    for i in range(opts['n_cycles']): # 50
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K) # random並び替え
        prec = np.zeros(K) # array
        # 動画シーケンスごとにループ
        for j,k in enumerate(k_list):
            tic = time.time()
            # pos_regions.shape([32,3,107,107])
            # neg_regions.shape([96,3,107,107])
            # pos_regions, neg_regions = dataset[k].next()
            # region(256)
            # shuffle
            dataset = dataset[k].next()
            #regions, labels = dataset[k].next()
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
            for a, b in loader:
                regions = a
                labels = b
            regions = Variable(regions)
            labels = Variable(labels)
            #pos_regions = Variable(pos_regions)
            #neg_regions = Variable(neg_regions)
        
            if opts['use_gpu']:
            #    pos_regions = pos_regions.cuda()
            #    neg_regions = neg_regions.cuda()
                 regions = regions.cuda()

        
            output = model(regions, k)
            #neg_score = model(neg_regions, k)

            # 損失計算
            #loss = criterion(pos_score, neg_score)
            loss = criterion(output, labels)
            # 勾配初期化
            model.zero_grad()
            loss.backward()
            # norm計算
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip']) # Maxnorm 10
            # パラメータ更新
            optimizer.step()
            
            #prec[k] = evaluator(pos_score, neg_score)
            prec[k] = evaluator(output, labels)

            toc = time.time()-tic
            print "Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                    (i, j, k, loss.data[0], prec[k], toc)

        cur_prec = prec.mean()
        print "Mean Precision: %.3f" % (cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print "Save model to %s" % opts['model_path']
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    train_mdnet()

