# -*- coding:utf-8 -*-
import os
import json
import numpy as np

def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name

        seq_home = '../dataset/pets2017'
        save_home = '../result_fig'
        result_home = '../result'
        
        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_1_path = os.path.join(seq_home, seq_name, 'gt_1.txt')
        gt_2_path = os.path.join(seq_home, seq_name, 'gt_2.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir,x) for x in img_list]

        # gt_1 = np.loadtxt(gt_1_path,delimiter=',')
        # gt_2 = np.loadtxt(gt_2_path,delimiter=',')
        gt_1 = None
        gt_2 = None
        # 最初1フレームのみ与える
        # init_bbox_1 = gt_1[0]
        # init_bbox_2 = gt_2[0]
        init_bbox_1 = np.loadtxt(gt_1_path,delimiter=',')[0]
        init_bbox_2 = np.loadtxt(gt_2_path,delimiter=',')[0]
        
        savefig_dir = os.path.join(save_home,seq_name)
        result_dir = os.path.join(result_home,seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir,'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json,'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None
        
    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox_1, init_bbox_2, gt_1, gt_2, savefig_dir, args.display, result_path
