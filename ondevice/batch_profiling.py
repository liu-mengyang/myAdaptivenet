import argparse
import os

import logging
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Subset
import numpy as np
import torch
import time
import torchvision
from mytimm.models import create_model
from mytimm.utils import *
from tools.evolution_finder import EvolutionFinder
import torchvision.transforms as transforms
import timm
from tools import mytools
from tools import global_var
from loguru import logger
import sys
import json

from utils import evaluate_latency, evaluate_accuracy

parser = argparse.ArgumentParser(description='evolution finder', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR', default='ValForDevice/ValForDevice3k',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='imagenet',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')

parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.05, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
#parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   # help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

parser.add_argument('--model_path', metavar='DIR',
                    help='path to trained model')

parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--GPU', action='store_true', default=True,
                    help='Use GPU')
parser.add_argument("--log_interval", default=200, type=int)
parser.add_argument("--warmupbatches", default=10, type=int)
parser.add_argument('--pths_path', metavar='DIR',
                    help='path to trained model')
parser.add_argument('--slim', action='store_true', default=False)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument("--batch_size_for_lat", default=4, type=int)
parser.add_argument('--pruned', action='store_true', default=False)
parser.add_argument('--save_path', default='./npy/batch-', type=str)
parser.add_argument('--baseline_save_path', default='./npy/baseline-batch-', type=str)
parser.add_argument("--num_workers", default=8, type=int)


parser.add_argument("--data_len", default=500, type=int)
parser.add_argument("--pre_len", default=3, type=int) 
parser.add_argument("--searching_times", default=10, type=int)
parser.add_argument("--population_size", default=200, type=int)

parser.add_argument("--load_times", default=2, type=int) 
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--baseline_load_times", default=1, type=int)
parser.add_argument("--baseline_batch_size", default=500, type=int)
parser.add_argument('--method', default='AdaptiveNet',type=str)
parser.add_argument('--time-budget', default=40, type=float)
parser.add_argument('--framework', default='ncnn', type=str)
parser.add_argument('--device', default='samsung1', type=str)
parser.add_argument("--command", default="ncnn_cpu_f0_fp16", type=str)
parser.add_argument("--range", nargs="+", default=None)
parser.add_argument("--benchmark", action='store_true', default=False)
parser.add_argument("--subnet_save_path", type=str, default=None)
    


args = parser.parse_args()

print("""Usage:
python3 batch_profiling.py \\
    --model {resnet50|mbv2} \\
    --framework {tflite|ncnn} \\
    --device {samsung0|samsung1|redmi0|honor0} \\
    --command {command} \\
    --subnet_save_path {subnets path}""")

if args.model == 'resnet50':
    args.model_path = 'weights/resnet1epoch59acc69.pth'
elif args.model == 'mbv2':
    args.model = 'mobilenetv2_100'
    args.model_path = 'weights/mbv2_100_1epoch44_best.pth'
else:
    raise ValueError("args.model")

if 'samsung' in args.device:
    subcommand = 'f0'
else:
    subcommand = '70'

if args.framework == 'ncnn':
    BACKEND_CONFIG = f'{args.device}'
    COMMAND_CONFIG = f'{args.command}'
    EVALUATE_CONFIG = 'im1k_gpu0_128'
elif args.framework == 'tflite':
    BACKEND_CONFIG = f'{args.device}'
    COMMAND_CONFIG = f'{args.command}'
    EVALUATE_CONFIG = 'im1k_gpu0_128'
else:
    raise ValueError("args.framework")

print('BACKEND_CONFIG', BACKEND_CONFIG)
print('COMMAND_CONFIG', COMMAND_CONFIG)



def main():
    logger.add(f"{args.model}_profiling.txt")
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    args.world_size = 1
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=1000,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if 'resnet' in args.model:
        model.get_skip_blocks(more_options=False)
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = mytools.prune_model(teachermodel, [0.2, 0.4])
            model.get_pruned_module(prune_modules)
            del teachermodel
    if 'mobilenetv2' in args.model:
        model.get_multi_blocks()
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = mytools.prune_mbv2(teachermodel, [0.25, 0.5])
            model.get_pruned_module(prune_modules)
            del teachermodel

    global_var._init()
    global_var.set_value('validated_feature', set())
    global_var.set_value('need_save_feature', set())
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=True)
    if args.method == "AdaptiveNet":
        mytools.load_data(model,args.model,args.data_dir,args.save_path, method = args.method, load_times=args.load_times,batch_size=args.batch_size,data_len=args.data_len)
    elif args.method == "BaseLine0":
        mytools.load_data(model,args.model,args.data_dir,args.baseline_save_path, method = args.method, load_times=args.load_times,batch_size=args.batch_size,data_len=args.data_len)
    
    if args.slim:
        model.adjust_channels()
        model.get_multi_blocks()
    elif 'mobilenetv2' in args.model:
        layer_lens = []
        for layeridx in range(len(model.multiblocks)):
            layerlen = len(model.multiblocks[layeridx])
            for blockidx in range(len(model.multiblocks[layeridx])): # note that we did not prune the last block
                layer_lens.append(layerlen)
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules, prune_checkpoint, prune_block_nums = mytools.prune_model_v2(teachermodel, [0.25, 0.5])
            model.get_pruned_module(prune_modules, prune_checkpoint, prune_block_nums)
            del teachermodel

    with open(args.subnet_save_path, 'r') as f:
        subnets = json.load(f)

    for constraint, subnet in subnets.items():
        model.apply_subnet(subnet['subnet'])
        best_latency = evaluate_latency(model, (1, 3, 224, 224), BACKEND_CONFIG, COMMAND_CONFIG)
        subnet['latency'] = best_latency*1000

        logger.info(f"Constraint: {constraint}, latency: {best_latency*1000}, accuracy {subnet['accuracy']}\n")
    
    with open(args.subnet_save_path, 'w') as f:
        json.dump(subnets, f, indent=4)


if __name__ == '__main__':
    main()
