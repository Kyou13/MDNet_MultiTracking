# -*- coding:utf-8 -*-
# my-py-MDNet/draw_img.py 描写
# my-py-MDNet/dataset/2DMOT2015/train/mdNet.py ラベル生成

# result.jsonとdatasetをそのままconvert配下に移動
# pymot/convert/converter_multiple.py 計測前のjsonデータ生成
# pymot/pymot.py 計測
# pymot.py -a
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
from sample_generator_v3 import *
from data_prov import *
from model_v3 import *
from bbreg import *
from options_v3 import *
from gen_config_v3 import *


# 乱数生成器初期化
np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    # ./data_prov
    global feats
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

# targetsは最大ID
def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, active_targets, targets, labels_=None, in_layer='fc4'):
    # set training mode
    model.train()
    
    batch_pos = opts['batch_pos'] * len(active_targets) # 32
    batch_neg = opts['batch_neg'] * len(active_targets) # 96
    batch_test = opts['batch_test'] * len(active_targets) # 256
    # このなかからハードネガティブマイニングを行う
    batch_neg_cand = max(opts['batch_neg_cand']*len(active_targets) , batch_neg) # 2048

    # ランダム並び替え
    pos_idx = np.random.permutation(pos_feats.size(0)) # len(3000)
    neg_idx = np.random.permutation(neg_feats.size(0)) # len(30000)
    # pos_idx拡張
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))]) # len(3000)
    while(len(neg_idx) < batch_neg_cand*maxiter): # 1024 * 30                         # len(30000)
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
            # 256ずつ回すneg_cand(1024)まで
            for start in range(0,batch_neg_cand,batch_test):
                # 256
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                # 初回ループ
                # その他の出力をとりだす
                if start==0:
                    neg_cand_score = score.data[:,targets].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,targets].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg) # 大きい96
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # len(11000)
        # postive dataのみのlabel
	# if labels is None: 
	if labels_ is not None:
	    labels = labels_

	else:
	    labels = np.array([])
	    for i in active_targets:
	        # 初期フレームpos_feats=3000
	        labels = np.append(labels,[i]*(len(pos_feats)//len(active_targets)))
	    labels = torch.LongTensor(labels)
        # top_idxはneg_cand_scoreのmaxと同じ
        
	# featsを合体
        _feats = torch.cat([batch_pos_feats,batch_neg_feats],0)
        # labels作成、その他を加える
        _labels = Variable(torch.cat([labels.index_select(0,pos_cur_idx.cpu()),torch.LongTensor([targets]*len(batch_neg_feats))],dim=0))
        # shuffle
        shuffle_idx = Variable(torch.LongTensor(np.random.permutation(len(_feats))))
        feats = torch.index_select(_feats, 0, shuffle_idx.cuda())
        labels = torch.index_select(_labels, 0, shuffle_idx)
        # forward
        # 128 * 3 , feats 128*4608
        score = model(feats, in_layer=in_layer, )

	if opts['use_gpu']:
            labels = labels.cuda()
        
        # optimize
        # neg,posのlog_softmaxをとり,足したものp
        loss = criterion(score, labels)
        model.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):

    # gt_...の列数
    zigen = 4
    # Init bbox
    target_bbox = np.zeros((len(init_bbox),zigen),float)
    # target_bbox = []

    # [[first_frame, x, y, width,height],
    # ...]]
    # 長さはtarget数と同じ
    print("ターゲット数:{}".format(len(init_bbox)))
    for i in range(len(init_bbox)):
	if init_bbox[i][2] != 0:
            target_bbox[i] = init_bbox[i]

    ## len(taget_bbox) == len(init_bbox)
    result = np.zeros((len(init_bbox),len(img_list),zigen))
    result_bb = np.zeros((len(init_bbox),len(img_list),zigen))

    # 1フレーム目に存在するターゲットID
    active_targets = []
    for i in range(len(init_bbox)):
	if gt[i][0][2] != 0:
	    active_targets.append(i)    
	    
    # 初期フレームのbboxを代入
    c = 0
    for i in range(len(init_bbox)):
	if i in active_targets:
            result[i][0] = target_bbox[i]
            result_bb[i][0] = target_bbox[i]
        
    # Init model
    model = MDNet(opts['model_path'], M=len(init_bbox)+1)
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
    bbreg_examples = []
    for i in range(len(target_bbox)):
	if target_bbox[i][2] != 0:
	    gen_samples_ = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                             			target_bbox[i],  opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'], i=0)
	    if len(gen_samples_) != 0:
                bbreg_examples.append(gen_samples_)
	    else:
		bbreg_examples.append([])
	else:
	    bbreg_examples.append([])

    bbreg_feats = []
    for i in range(len(target_bbox)):
	if target_bbox[i][2] != 0 and len(bbreg_examples[i]) != 0:
	    bbreg_feats.append(forward_samples(model, image, bbreg_examples[i]))
	else:
	    bbreg_feats.append([])

    bbreg = []
    for i in range(len(target_bbox)):
	if target_bbox[i][2] != 0 and len(bbreg_examples[i]) != 0:
	    bbreg.append(BBRegressor(image.size))
	else:
	    bbreg.append([])
    
    for i in range(len(target_bbox)):
	if target_bbox[i][2] != 0 and len(bbreg_examples[i]) != 0:
	    bbreg[i].train(bbreg_feats[i], bbreg_examples[i], target_bbox[i])

    # Draw pos/neg samples

    pos_examples = []
    # for i in range(len(init_bbox)):
    for i in range(len(target_bbox)):
	if target_bbox[i][2] != 0:
            pos_examples.append(gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                                              target_bbox[i], opts['n_pos_init'], opts['overlap_pos_init']))
    # neg
    # len(10000)
    _neg_examples = []
    for i in range(len(target_bbox)):
        # 処理対象以外のbboxを取り出す
	if target_bbox[i][2] != 0:
            other_bbox = [target_bbox[j] for j in range(len(target_bbox)) if j != i and target_bbox[j][2] != 0]
            _neg_examples.append(np.concatenate([
                            gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                        target_bbox[i], opts['n_neg_init']//2, opts['overlap_neg_init'], other_bbox=other_bbox, i=0),
                            gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                        target_bbox[i], opts['n_neg_init']//2, opts['overlap_neg_init'], other_bbox=other_bbox, i=0)]))


    neg_examples = []
    for i in range(len(active_targets)):
        neg_examples.append(np.random.permutation(_neg_examples[i]))

    # 途中までフォワード
    # 500 * 4608
    pos_feats = []
    for i in range(len(active_targets)): 
    	pos_feats.append(forward_samples(model, image, pos_examples[i]))

    # 5000 * 4608
    neg_feats = []
    for i in range(len(active_targets)): 
    	neg_feats.append(forward_samples(model, image, neg_examples[i]))

    # int(4608)
    feat_dim = pos_feats[0].size(-1)

    first_flag = False 
    for i in range(len(active_targets)):
	# ループの最初は代入、それ以降はconcat
        if first_flag == False:
	    pos_feats_cat = pos_feats[i] 
            neg_feats_cat = neg_feats[i]
            first_flag = True
	else:
	    pos_feats_cat = torch.cat((pos_feats_cat,pos_feats[i]),0)    
	    neg_feats_cat = torch.cat((neg_feats_cat,neg_feats[i]),0)    

    # Initial training # maxiter_init:30
    train(model, criterion, init_optimizer, pos_feats_cat, neg_feats_cat, opts['maxiter_init'], active_targets,len(init_bbox))
    
    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # TODO
    # 50 * 4608
    pos_feats_all_ = []
    for i in range(len(init_bbox)):
	pos_feats_all_.append([])
    for (c,i) in enumerate(active_targets): 
	#if i in active_targets:
	pos_feats_all_[i] = pos_feats[c][:opts['n_pos_update']]
    pos_feats_all = [pos_feats_all_]    
    
        
    # 200 * 4608
    neg_feats_all_ = []
    for i in range(len(init_bbox)):
	neg_feats_all_.append([])
    for (c,i) in enumerate(active_targets): 
	neg_feats_all_[i] = neg_feats[c][:opts['n_neg_update']]

    neg_feats_all = [neg_feats_all_]    
    
    # active_targets_all = [active_targets]
    spf_total = time.time()-tic
    
    # これまで出たターゲットを保存
    all_active_targets = active_targets
    print(active_targets)
    
    # Main loop
    # -----------------------------------
    # | gtをみて、いる、いないは与える！|
    # -----------------------------------
    for i in range(1,len(img_list)):
    # targetは毎フレーム定義される

        tic = time.time()

        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # old_active_targets = active_targets

	# gt.shape(targets, frames, 4)
	# 現フレームに存在するtargetIDを取り出す
        active_targets = []
        for j in range(len(init_bbox)):
    	    if gt[j][i][2] != 0:
    	        active_targets.append(j)    
        print("active targets:{}".format(active_targets))
	# ID保存している
	# active_targets_all.append(active_targets)
        # 新しく出現したターゲットいないか確認する
        train_frag = False   
	# 新しく出現したIDを格納
	## TODO 前フレームとしか比較していない
	new_targets=[]
        for j in active_targets:
	    if j not in all_active_targets:
                train_frag = True   
		new_targets.append(j)
		all_active_targets.append(j)
	
	# 新しく出た奴がいた
	if train_frag == True:
	    print("\nchange_active_targets")
	    # 共通のID
	    common_idx = []
            for j in active_targets:
	        if j not in new_targets:
	      	    common_idx.append(j) 
	    # new_target_bboxは現フレームのbbox
	    # target_bboxは前のフレームのbboxが格納されている
			
	    # active_targets = common_targets
            # active_targets.extend([j for j in new_targets])

	    for j in new_targets:
		target_bbox[j] = gt[j][i]
	
	    # IDを保存する
	    # active_targets = [j for j in common_targets]
            # active_targets.extend([j for j in new_targets])
		
    	    pos_examples = []
    	    for j in range(len(active_targets)):
    	        pos_examples.append(gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
    	                                          target_bbox[active_targets[j]], opts['n_pos_init'], opts['overlap_pos_init']))
    	    _neg_examples = []
    	    for j in range(len(active_targets)):
    	        other_bbox = [target_bbox[k] for k in range(len(target_bbox)) if k != j and target_bbox[k][2] != 0]
    	        _neg_examples.append(np.concatenate([
    	                            gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
    	                                        target_bbox[active_targets[j]], opts['n_neg_init']//2, opts['overlap_neg_init'], other_bbox=other_bbox),
    	                            gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
    	                                        target_bbox[active_targets[j]], opts['n_neg_init']//2, opts['overlap_neg_init'], other_bbox=other_bbox)]))

    	    neg_examples = []
    	    for j in range(len(active_targets)):
    	        neg_examples.append(np.random.permutation(_neg_examples[j]))
    	    # Extract pos/neg features
    	    # 500 * 4608
    	    pos_feats = []
    	    for j in range(len(active_targets)): 
    	    	pos_feats.append(forward_samples(model, image, pos_examples[j]))

    	    # 5000 * 4608
    	    neg_feats = []
    	    for j in range(len(active_targets)): 
    	    	neg_feats.append(forward_samples(model, image, neg_examples[j]))

    	    # int(4608)
    	    feat_dim = pos_feats[0].size(-1)

    	   
    	    first_flag = False 
    	    for j in range(len(active_targets)):
    	        if first_flag == False:
    	            pos_feats_cat = pos_feats[j] 
    	            neg_feats_cat = neg_feats[j]
    	            first_flag = True
    	        else:
    	            pos_feats_cat = torch.cat((pos_feats_cat,pos_feats[j]),0)    
    	            neg_feats_cat = torch.cat((neg_feats_cat,neg_feats[j]),0)    

    	    train(model, criterion, init_optimizer, pos_feats_cat, neg_feats_cat, opts['maxiter_update'], active_targets,len(init_bbox))

	    pos_feats_all_ = []
	    for j in range(len(init_bbox)):
		pos_feats_all_.append([])		
	    for (c,j) in enumerate(active_targets):
		pos_feats_all_[j] = pos_feats[c][:opts['n_pos_update']]
	    pos_feats_all.append(pos_feats_all_)

	    neg_feats_all_ = []
	    for j in range(len(init_bbox)):
		neg_feats_all_.append([])		
	    for (c,j) in enumerate(active_targets):
		neg_feats_all_[j] = neg_feats[c][:opts['n_neg_update']]
	    neg_feats_all.append(neg_feats_all_)

            ## 04/29
            if len(pos_feats_all) > opts['n_frames_long']: # 20
                del pos_feats_all[0]

            if len(neg_feats_all) > opts['n_frames_short']: # 20
                del neg_feats_all[0]
                # del pos_feats_all[0]


	    bbreg_examples = []
	    for j in range(len(target_bbox)):
		if target_bbox[j][2] != 0:
                    gen_samples_ = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
	                             			target_bbox[j],  opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'],i=0)
		    if len(gen_samples_) != 0:
	                bbreg_examples.append(gen_samples_)
		    else:
			bbreg_examples.append([])
		else:
		    bbreg_examples.append([])

	    bbreg_feats = []
	    for j in range(len(target_bbox)):
		if target_bbox[j][2] != 0 and len(bbreg_examples[j]) != 0:
		    bbreg_feats.append(forward_samples(model, image, bbreg_examples[j]))
		else:
		    bbreg_feats.append([])
	
	    bbreg = []
	    for j in range(len(target_bbox)):
		if target_bbox[j][2] != 0 and len(bbreg_examples[j]) != 0:
		    bbreg.append(BBRegressor(image.size))
		else:
		    bbreg.append([])
	    
	    for j in range(len(target_bbox)):
		if target_bbox[j][2] != 0 and len(bbreg_examples[j]) != 0:
		    bbreg[j].train(bbreg_feats[j], bbreg_examples[j], target_bbox[j])
            # target_bbox = new_target_bbox
            bbreg_bbox = target_bbox
		
	    for j in active_targets:
		result[j][i] = target_bbox[j]
		result_bb[j][i] = bbreg_bbox[j]


	# 新しいbboxが出現しなかった
	else:
        # Estimate target bbox
        # 256 * 4
	    samples = []
	    ## len(target_bbox == init_bbox
	    # active_targetsにはIDが含まれている
	    for j in active_targets:
	    	samples.append(gen_samples(sample_generator, target_bbox[j], opts['n_samples']))
            
	    # --------
	    # forward
	    # --------
            # 256 * targets+1
            sample_scores = []
	    for j in range(len(active_targets)):
	        sample_scores.append(forward_samples(model, image, samples[j], out_layer='fc6'))
            # high score top5
            # 出力のうち
            comp_high = []
            for j in range(len(sample_scores)):
                comp_high_temp = torch.ones(256).type(torch.ByteTensor)
                if opts['use_gpu']:
                    comp_high_temp = comp_high_temp.cuda()
                for k in range(len(sample_scores)):
                    if j != k:
                        comp_high_temp = (sample_scores[j][:,active_targets[j]] > sample_scores[j][:,active_targets[k]]) == comp_high_temp

                comp_high.append(comp_high_temp) # len(active_targets)

            # ほんとに0でいい？
	    # TODO
            for j in range(len(active_targets)):
                for c,comp in enumerate(comp_high[j]):
                    if not comp:
                        sample_scores[j][c,active_targets[j]] = 0

	    ## TODO ないターゲットに対する処理は必要か？

            top_scores = []
            for j in range(len(active_targets)):
                top_scores.append(sample_scores[j][:,active_targets[j]].topk(5))

            # numpy配列へ変換
            top_idx = []
            target_score = []
            target_bbox = []
            # TODO:対象外のtarget
            for j in range(len(active_targets)):
                target_score.append(top_scores[j][0].mean())
                top_idx.append(top_scores[j][1].cpu().numpy())
                # target_bbox.append(samples[j][top_scores[j][1].cpu().numpy()].mean(axis=0))
	        target_bbox.append(samples[j][top_idx[j]].mean(axis=0))
            
	    # target_bbox,target_scoreが0のものは
            
	    target_score = np.array(target_score)
	    # ないターゲットは除く
	    success = min(target_score) > opts['success_thr']

            
            # Expand search area at failure
            # スコア悪かったら座標にかける定数変更
            if success:
                sample_generator.set_trans_f(opts['trans_f']) # 0.6
            else:
                sample_generator.set_trans_f(opts['trans_f_expand']) # 1.5

            # Bbox regression
            # out 03/24
	    target_bbox_ = target_bbox
	    target_bbox = np.zeros((len(init_bbox),4))
	    for (c,j) in enumerate(active_targets):
	        target_bbox[j] = target_bbox_[c]

	    # if success:
    	    #     bbreg_bbox = np.zeros((len(init_bbox),zigen),float)
	    #     for j in range(len(active_targets)):
	    #         if type(bbreg[active_targets[j]]) is not list:
            #             bbreg_samples = samples[j][top_idx[j]]
	    #             bbreg_feats = forward_samples(model, image, bbreg_samples)
	    #             bbreg_samples = bbreg[active_targets[j]].predict(bbreg_feats, bbreg_samples)
	    #             bbreg_bbox[active_targets[j]] = bbreg_samples.mean(axis=0)
	    #         else:
	    #     	bbreg_bbox[active_targets[j]] = target_bbox[active_targets[j]]
	    # else:
            bbreg_bbox = target_bbox

	    ## 新しく出現した時の処理をここに書く

	    # TODO 精度がわるくなったらの部分は後で

            # Copy previous result at failure
            if not success:
                for j in active_targets:
                    target_bbox[j] = result[j][i-1]
                    bbreg_bbox[j] = result[j][i-1]
            
            # Save result
            for j in active_targets:
                result[j][i] = target_bbox[j]
                result_bb[j][i] = bbreg_bbox[j]

            # Data collect
            # if success:
            # Draw pos/neg samples
            # 今回の予測で得られたbbox
            pos_examples = []
            neg_examples = []
	    for j in active_targets:
                pos_examples.append(gen_samples(pos_generator, target_bbox[j], 
                                           opts['n_pos_update'], # 50
                                           opts['overlap_pos_update'])) # [0.7, 1]

            	other_bbox = [target_bbox[k] for k in range(len(target_bbox)) if k != j and target_bbox[k][2] != 0]
                neg_examples.append(gen_samples(neg_generator, target_bbox[j],
                                               opts['n_neg_update'], # 200
                                               opts['overlap_neg_update'], # [0, 0.3]
                                               other_bbox=other_bbox)) 

            # Extract pos/neg features
            # conv3の出力
    	    pos_feats = []
    	    neg_feats = []
    	    for j in range(len(active_targets)): 
    	        pos_feats.append(forward_samples(model, image, pos_examples[j]))
    	        neg_feats.append(forward_samples(model, image, neg_examples[j]))

            # axis=0にappend
            # listの中に,50*4608のFloatTensorをappend
	    pos_feats_all_ = []
	    neg_feats_all_ = []
	    for j in range(len(init_bbox)):
		pos_feats_all_.append([])
		neg_feats_all_.append([])

	    for (c,j) in enumerate(active_targets):
	    	pos_feats_all_[j] = pos_feats[c]
            	# axis=0にappend
	    	neg_feats_all_[j] = neg_feats[c]

	    pos_feats_all.append(pos_feats_all_)
	    neg_feats_all.append(neg_feats_all_)
            ## 04/29
            if len(pos_feats_all) > opts['n_frames_long']: # 100
                # del neg_feats_all[0]
                del pos_feats_all[0]

	    # TODO 
            if len(neg_feats_all) > opts['n_frames_short']: # 20
                del neg_feats_all[0]

            # Short term update
            # 精度が悪くなったら
            # if i % opts['long_interval'] == 0:
            # if not success:
	    #     
	    #     print("\nretrain")
            #     # feat_dim = 4608
	    #     # 20
            #     nframes = min(opts['n_frames_short'],len(pos_feats_all))
            #     # axis = 0
            #     # view -1 はあわせる
            #     # pos_data_0 = torch.stack(pos_feats_all_0[-nframes:],0).view(-1,feat_dim)
            #     # neg_data_0 = torch.stack(neg_feats_all_0,0).view(-1,feat_dim)

	    #     # gt.shape (8,71,4)
	    #     # ターゲットごとに
	    #    	labels = np.array([])
	    #     for j in range(len(active_targets)):
	    #         pos_data_ = []
	    #         neg_data_ = []

	    #         l = 0
	    #         if i > opts['n_frames_short']:
	    #             l = i-opts['n_frames_short']
	    #         for k in range(nframes):
	    # 	    # if gt[j][len(pos_feats_all)-k][2] != 0:
	    # 	        if gt[active_targets[j]][l][2] != 0:
	    # 	            pos_data_.append(pos_feats_all[len(pos_feats_all)-nframes+k][active_targets[j]])
	    # 	            neg_data_.append(neg_feats_all[k][active_targets[j]])
	    #     	l = l+1
	    #         try:
            #             pos_data_ = torch.stack(pos_data_,0).view(-1,feat_dim)
	    #         except:
	    #     	import pdb;pdb.set_trace()
            #         neg_data_ = torch.stack(neg_data_,0).view(-1,feat_dim)

	    #         labels = np.append(labels,[active_targets[j]]*pos_data_.size()[0])
	    #    	    labels = torch.LongTensor(labels)
	    # 	    if j==0:
	    # 	        pos_data = pos_data_
	    # 	        neg_data = neg_data_
	    #         else:
	    # 	        pos_data = torch.cat((pos_data,pos_data_),0)
	    # 	        neg_data = torch.cat((neg_data,neg_data_),0)
	    # 	
            #     train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], active_targets,len(init_bbox),labels)
            
            # Long term update
            # 10frameおき
            if i % opts['long_interval'] == 0:
	       	labels = np.array([])
	        for j in range(len(active_targets)): ## 6
	            pos_data_ = []
	            neg_data_ = []
	            for k in range(len(pos_feats_all)):
	    	        #if gt[active_targets[j]][l][2] != 0:
	    	        if len(pos_feats_all[k][active_targets[j]]) != 0:
	        	    # pos_feats_allは最長20
	    	            pos_data_.append(pos_feats_all[k][active_targets[j]])

	            for k in range(len(neg_feats_all)):
	    	        #if gt[active_targets[j]][l][2] != 0:
	    	        if len(neg_feats_all[k][active_targets[j]]) != 0:
	        	    # pos_feats_allは最長20
	    	            neg_data_.append(neg_feats_all[k][active_targets[j]])

	            pos_data_ = torch.stack(pos_data_).view(-1,feat_dim)
	            neg_data_ = torch.stack(neg_data_).view(-1,feat_dim)

	            # ターゲット数過去つねにいるわけではないので
	            labels = np.append(labels,[active_targets[j]]*pos_data_.size()[0])
	       	    labels = torch.LongTensor(labels)
	            # 最初のID
	    	    if j==0:
	    	        pos_data = pos_data_
	    	        neg_data = neg_data_
	            else:
	        	# TODO ここ次元が合わないこともあるはず
	    	        pos_data = torch.cat((pos_data,pos_data_),0)
	    	        neg_data = torch.cat((neg_data,neg_data_),0)
    	        train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], active_targets,len(init_bbox),labels) #最後のやつはその他のID
            
            spf = time.time()-tic
            spf_total += spf

	    # 結果出力
            print "Frame %d/%d spf:%d" % (i+1,len(img_list),spf)
     	    for j in range(len(active_targets)):
	        print('overlap%d: %.3f,'%(active_targets[j],overlap_ratio(gt[active_targets[j]][i],result_bb[active_targets[j]][i])[0]))
	        print('target_score%d: %.3f,'%(active_targets[j],target_score[j]))
    	    res = {}
    	    for j in range(len(init_bbox)):
    	        res['res{}'.format(j)] = result_bb[j].round().tolist()
            json.dump(res, open('../result/dev_multiple/tmp.json', 'w'), indent=2)
		    

    fps = len(img_list) / spf_total
    spf = spf_total / len(img_list)
    # return result_1, result_2, result_3, result_1_bb, result_2_bb, result_3_bb, fps
    return result, result_bb, fps, spf

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    
    args = parser.parse_args()
    assert(args.seq != '' or args.json != '')
    
    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
    # img_list = img_list[:10]
    # Run tracker
    # result_1, result_2, result_3, result_1_bb, result_2_bb, result_3_bb, fps = run_mdnet(img_list, init_bbox, gt, savefig_dir=savefig_dir, display=display)
    result, result_bb, fps, spf = run_mdnet(img_list, init_bbox, gt, savefig_dir=savefig_dir, display=display)
    
    # Save result
    res = {}
    # res['res2'] = result_2_bb.round().tolist()
    # res['res3'] = result_3_bb.round().tolist()
    # res['type'] = 'rect'
    # res['fps'] = fps
    for i in range(len(init_bbox)):
        res['res{}'.format(i)] = result_bb[i].round().tolist()

    res['type'] = 'rect'
    res['fps'] = fps
    res['spf'] = spf
    json.dump(res, open(result_path, 'w'), indent=2)
