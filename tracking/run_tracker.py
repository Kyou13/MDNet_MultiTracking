# -*- coding:utf-8 -*-
import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *

# 乱数生成器初期化
np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    # ./data_prov
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        # (len(samples) ,3,107,107)
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.data.clone()
        else:
            # 行方向に連結
            feats = torch.cat((feats,feat.data.clone()),0)
    return feats


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


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    # set training mode
    model.train()
    
    batch_pos = opts['batch_pos'] # 32
    batch_neg = opts['batch_neg'] # 96
    batch_test = opts['batch_test'] #256
    # このなかからハードネガティブマイニングを行う
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg) # 10240

    # ランダム並び替え
    pos_idx = np.random.permutation(pos_feats.size(0)) # len(1000)
    neg_idx = np.random.permutation(neg_feats.size(0)) # len(10000)
    # pos_idx拡張
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter): # 10240 * 30
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    # default:30
    for iter in range(maxiter): 
        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long() # casts
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next] # 10240ずつ
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        # torch.index_select(input, dim, index, out=None) 
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        # neg scoreおおきいものをbatch_neg個とりだす
        if batch_neg_cand > batch_neg:# 96
            model.eval()
            # 256ずつ回す10240まで
            for start in range(0,batch_neg_cand,batch_test):
                # 256
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                # 初回ループ
                if start==0:
                    neg_cand_score = score.data[:,2].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,2].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg) # 大きい192
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # len(11000)
        labels = torch.LongTensor(np.concatenate(([0]*(len(pos_feats)/2),[1]*(len(pos_feats)/2),[2]*len(neg_feats))))
        # negativeラベルに1000のオフセットを持たせる 
        label_neg_idx = top_idx + torch.LongTensor([1000]*batch_neg).cuda()
        # 256(64+192) * 4608
        ###########################
        ## ランダムで入力シャッフル
        ###########################
        
        _feats = torch.cat([batch_pos_feats,batch_neg_feats],0)
        _labels = Variable(torch.cat([labels.index_select(0,pos_cur_idx),labels.index_select(0,label_neg_idx)],dim=0))
        feats = _feats[0::(len(_feats)/2)]
        labels = _labels[0::(len(_labels)/2)]
        for i in range(1,(len(_feats)/2)):
            _tmp_feats = _feats[i::(len(_feats)/2)]
            feats = torch.cat([feats,_tmp_feats],0)
            _tmp_labels = _labels[i::(len(_labels)/2)]
            labels = torch.cat([labels,_tmp_labels],0)
        # forward
        # 128 * 3 , feats 128*4608
        score = model(feats, in_layer=in_layer)
        
        # optimize
        # neg,posのlog_softmaxをとり,足したものp
        loss = criterion(score, labels)
        model.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        #print "Iter %d, Loss %.4f" % (iter, loss.data[0])


def run_mdnet(img_list, init_bbox_1, init_bbox_2, gt_1=None, gt_2=None, savefig_dir='', display=False):

    # Init bbox
    target_bbox_0 = np.array(init_bbox_1)
    target_bbox_1 = np.array(init_bbox_2)
    result_1 = np.zeros((len(img_list),4))
    result_2 = np.zeros((len(img_list),4))
    result_1_bb = np.zeros((len(img_list),4))
    result_2_bb = np.zeros((len(img_list),4))
    result_1[0] = target_bbox_0
    result_2[0] = target_bbox_1
    result_1_bb[0] = target_bbox_0
    result_2_bb[0] = target_bbox_1

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    # fc層のみ勾配を計算
    model.set_learnable_params(opts['ft_layers']) # fc
    
    # Init criterion and optimizer 
    criterion = nn.CrossEntropyLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    # Train bbox regressor
    # sample_generator/gen_samples and SampleGenerator
    # len(927)
    # Overlapが閾値超えたやつを選択 1000個以下
    bbreg_1_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox_0, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_2_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox_1, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    # 927 * 4608(512*3*3) conv3の特徴量
    bbreg_1_feats = forward_samples(model, image, bbreg_1_examples)
    bbreg_2_feats = forward_samples(model, image, bbreg_2_examples)
    # ./bbreg.py
    bbreg_1 = BBRegressor(image.size)
    bbreg_1.train(bbreg_1_feats, bbreg_1_examples, target_bbox_0)
    bbreg_2 = BBRegressor(image.size)
    bbreg_2.train(bbreg_2_feats, bbreg_2_examples, target_bbox_1)

    # Draw pos/neg samples
    pos_1_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox_0, opts['n_pos_init'], opts['overlap_pos_init'])
    pos_2_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox_1, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_1_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                target_bbox_0, opts['n_neg_init']//2, opts['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                target_bbox_0, opts['n_neg_init']//2, opts['overlap_neg_init'])])
    neg_2_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                target_bbox_1, opts['n_neg_init']//2, opts['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                target_bbox_1, opts['n_neg_init']//2, opts['overlap_neg_init'])])
    neg_1_examples = np.random.permutation(neg_1_examples)
    neg_2_examples = np.random.permutation(neg_2_examples)

    # Extract pos/neg features
    # 500 * 4608
    pos_1_feats = forward_samples(model, image, pos_1_examples)
    pos_2_feats = forward_samples(model, image, pos_2_examples)
    # 5000 * 4608
    neg_1_feats = forward_samples(model, image, neg_1_examples)
    neg_2_feats = forward_samples(model, image, neg_2_examples)
    # int(4608)
    feat_dim = pos_1_feats.size(-1)

    # 1000 * 4608
    pos_feats = torch.cat((pos_1_feats,pos_2_feats),0)
    # 10000 * 4608
    neg_feats = torch.cat((neg_1_feats,neg_2_feats),0)

    # Initial training # maxiter_init:30
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    
    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    # 50 * 4608
    pos_feats_all_0 = [pos_1_feats[:opts['n_pos_update']]]
    pos_feats_all_1 = [pos_2_feats[:opts['n_pos_update']]]
    neg_feats_all_0 = [neg_1_feats[:opts['n_neg_update']]]
    neg_feats_all_1 = [neg_2_feats[:opts['n_neg_update']]]
    
    spf_total = time.time()-tic

    # Display
    # -f -d オプションあったら実行
    savefig = savefig_dir != ''
    if display or savefig: 
        dpi = 80.0
        figsize = (image.size[0]/dpi, image.size[1]/dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        # [left,top,width,height]
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #im = ax.imshow(image, aspect='normal')
        im = ax.imshow(image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0,:2]),gt[0,2],gt[0,3], 
                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        
        rect = plt.Rectangle(tuple(result_bb[0,:2]),result_bb[0,2],result_bb[0,3], 
                linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir,'0000.jpg'),dpi=dpi)
    
    # Main loop
    for i in range(1,len(img_list)):

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox
        # 256 * 4
        samples_0 = gen_samples(sample_generator, target_bbox_0, opts['n_samples'])
        samples_1 = gen_samples(sample_generator, target_bbox_1, opts['n_samples'])
        # 256 * 3
        sample_scores_0 = forward_samples(model, image, samples_0, out_layer='fc6')
        sample_scores_1 = forward_samples(model, image, samples_1, out_layer='fc6')
        # high score top5
        comp_high_0 = (sample_scores_0[:,0] > sample_scores_0[:,1]) == (sample_scores_0[:,0] > sample_scores_0[:,2])
        comp_high_1 = (sample_scores_1[:,1] > sample_scores_1[:,0]) == (sample_scores_1[:,1] > sample_scores_1[:,2])
        for c,(comp_0,comp_1) in enumerate(zip(comp_high_0, comp_high_1)):
            if not comp_0:
                sample_scores_0[c,0] = 0  
            if not comp_1:
                sample_scores_1[c,1] = 0  

        top_scores_0, top_idx_0 = sample_scores_0[:,0].topk(5)
        top_scores_1, top_idx_1 = sample_scores_1[:,1].topk(5)
        # numpy配列へ変換
        top_idx_0 = top_idx_0.cpu().numpy()
        top_idx_1 = top_idx_1.cpu().numpy()
        target_score_0 = top_scores_0.mean()
        target_score_1 = top_scores_1.mean()
        target_bbox_0 = samples_0[top_idx_0].mean(axis=0)
        target_bbox_1 = samples_1[top_idx_1].mean(axis=0)
        

        success = min(target_score_0, target_score_1) > opts['success_thr'] # 0
        
        # Expand search area at failure
        # スコア悪かったら範囲を広げる
        if success:
            sample_generator.set_trans_f(opts['trans_f']) # 0.6
        else:
            sample_generator.set_trans_f(opts['trans_f_expand']) # 1.5

        # Bbox regression
        if success:
            bbreg_samples_0 = samples[top_idx_0]
            bbreg_samples_1 = samples[top_idx_1]
            # 5 * 4608
            bbreg_feats_0 = forward_samples(model, image, bbreg_samples_0)
            bbreg_feats_1 = forward_samples(model, image, bbreg_samples_1)
            # 5 * 4
            bbreg_samples_0 = bbreg_1.predict(bbreg_feats_0, bbreg_samples_0)
            bbreg_samples_1 = bbreg_2.predict(bbreg_feats_1, bbreg_samples_1)
            # 1 * 4
            bbreg_bbox_0 = bbreg_samples_0.mean(axis=0)
            bbreg_bbox_1 = bbreg_samples_1.mean(axis=0)
        else:
            bbreg_bbox_0 = target_bbox_0
            bbreg_bbox_1 = target_bbox_1
        
        # Copy previous result at failure
        if not success:
            target_bbox_0 = result_1[i-1]
            target_bbox_1 = result_2[i-1]
            bbreg_bbox_0 = result_1_bb[i-1]
            bbreg_bbox_1 = result_2_bb[i-1]
        
        # Save result
        result_1[i] = target_bbox_0
        result_2[i] = target_bbox_1
        # とは
        result_1_bb[i] = bbreg_bbox_0
        result_2_bb[i] = bbreg_bbox_1

        # Data collect
        if success:
            # Draw pos/neg samples
            # 今回の予測で得られたbbox
            pos_examples_0 = gen_samples(pos_generator, target_bbox_0, 
                                       opts['n_pos_update'], # 50
                                       opts['overlap_pos_update']) # [0.7, 1]
            pos_examples_1 = gen_samples(pos_generator, target_bbox_1, 
                                       opts['n_pos_update'], # 50
                                       opts['overlap_pos_update']) # [0.7, 1]
            neg_examples_0 = gen_samples(neg_generator, target_bbox_0, 
                                       opts['n_neg_update'], # 200
                                       opts['overlap_neg_update']) # [0, 0.3]
            neg_examples_1 = gen_samples(neg_generator, target_bbox_1, 
                                       opts['n_neg_update'], # 200
                                       opts['overlap_neg_update']) # [0, 0.3]

            # Extract pos/neg features
            # 50 * 4608
            pos_feats_0 = forward_samples(model, image, pos_examples_0)
            pos_feats_1 = forward_samples(model, image, pos_examples_1)
            neg_feats_0 = forward_samples(model, image, neg_examples_0)
            neg_feats_1 = forward_samples(model, image, neg_examples_1)
            # axis=0にappend
            # listの中に,50*4608のFloatTensorをappend
            pos_feats_all_0.append(pos_feats_0)
            pos_feats_all_1.append(pos_feats_1)
            neg_feats_all_0.append(neg_feats_0)
            neg_feats_all_1.append(neg_feats_1)
            # 
            if len(pos_feats_all_0) > opts['n_frames_long']: # 100
                del pos_feats_all_0[0]
                del pos_feats_all_1[0]
            if len(neg_feats_all_1) > opts['n_frames_short']: # 20
                del neg_feats_all_0[0]
                del neg_feats_all_1[0]

        # Short term update
        # 精度が悪くなったら
        if not success:
            # feat_dim = 4608
            nframes = min(opts['n_frames_short'],len(pos_feats_all_0))
            # axis = 0
            # view -1 はあわせる
            pos_data_0 = torch.stack(pos_feats_all_0[-nframes:],0).view(-1,feat_dim)
            pos_data_1 = torch.stack(pos_feats_all_1[-nframes:],0).view(-1,feat_dim)
            neg_data_0 = torch.stack(neg_feats_all_0,0).view(-1,feat_dim)
            neg_data_1 = torch.stack(neg_feats_all_1,0).view(-1,feat_dim)
            pos_data = torch.cat((pos_data_0,pos_data_1),0)
            neg_data = torch.cat((neg_data_0,neg_data_1),0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        # Long term update
        # 10frameおき
        # 
        elif i % opts['long_interval'] == 0:
            pos_data_0 = torch.stack(pos_feats_all_0,0).view(-1,feat_dim)
            pos_data_1 = torch.stack(pos_feats_all_1,0).view(-1,feat_dim)
            neg_data_0 = torch.stack(neg_feats_all_0,0).view(-1,feat_dim)
            neg_data_1 = torch.stack(neg_feats_all_1,0).view(-1,feat_dim)
            pos_data = torch.cat((pos_data_0,pos_data_1),0)
            neg_data = torch.cat((neg_data_0,neg_data_1),0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        spf = time.time()-tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt_1 is not None:
                gt_rect.set_xy(gt[i,:2])
                gt_rect.set_width(gt[i,2])
                gt_rect.set_height(gt[i,3])

            rect.set_xy(result_bb[i,:2])
            rect.set_width(result_bb[i,2])
            rect.set_height(result_bb[i,3])
            
            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir,'%04d.jpg'%(i)),dpi=dpi)

        if gt is None:
            print "Frame %d/%d, Score0 %.3f, Score1 %.3f, Time %.3f" % \
                (i, len(img_list), target_score_0,target_score_1, spf)
        else:
            print "Frame %d/%d, Overlap0 %.3f, Overlap1 %.3f, Score0 %.3f, Score1 %.3f, Time %.3f" % \
                (i, len(img_list), overlap_ratio(gt_1[i],result_1_bb[i])[0], overlap_ratio(gt_2[i],result_2_bb[i])[0], target_score_0, target_score_1, spf)

    fps = len(img_list) / spf_total
    return result_1, result_2, result_1_bb, result_2_bb, fps


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    
    args = parser.parse_args()
    assert(args.seq != '' or args.json != '')
    
    # Generate sequence config
    img_list, init_bbox_1, init_bbox_2, gt_1, gt_2, savefig_dir, display, result_path = gen_config(args)

    # Run tracker
    result_1, result_2, result_1_bb, result_2_bb, fps = run_mdnet(img_list, init_bbox_1, init_bbox_2, gt_1=gt_1, gt_2=gt_2, savefig_dir=savefig_dir, display=display)
    
    # Save result
    res = {}
    res['res1'] = result_1_bb.round().tolist()
    res['res2'] = result_2_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)
