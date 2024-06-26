""" The EfficientNet Family in PyTorch

An implementation of EfficienNet that covers variety of related models with efficient architectures:

* EfficientNet-V2
  - `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298

* EfficientNet (B0-B8, L2 + Tensorflow pretrained AutoAug/RandAug/AdvProp/NoisyStudent weight ports)
  - EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
  - CondConv: Conditionally Parameterized Convolutions for Efficient Inference - https://arxiv.org/abs/1904.04971
  - Adversarial Examples Improve Image Recognition - https://arxiv.org/abs/1911.09665
  - Self-training with Noisy Student improves ImageNet classification - https://arxiv.org/abs/1911.04252

* MixNet (Small, Medium, and Large)
  - MixConv: Mixed Depthwise Convolutional Kernels - https://arxiv.org/abs/1907.09595

* MNasNet B1, A1 (SE), Small
  - MnasNet: Platform-Aware Neural Architecture Search for Mobile - https://arxiv.org/abs/1807.11626

* FBNet-C
  - FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS - https://arxiv.org/abs/1812.03443

* Single-Path NAS Pixel1
  - Single-Path NAS: Designing Hardware-Efficient ConvNets - https://arxiv.org/abs/1904.02877

* TinyNet
    - Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets - https://arxiv.org/abs/2010.14819
    - Definitions & weights borrowed from https://github.com/huawei-noah/CV-Backbones/tree/master/tinynet_pytorch

* And likely more...

The majority of the above models (EfficientNet*, MixNet, MnasNet) and original weights were made available
by Mingxing Tan, Quoc Le, and other members of their Google Brain team. Thanks for consistently releasing
the models and weights open source!

Hacked together by / Copyright 2019, Ross Wightman
"""
import copy
import sys
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mytimm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .efficientnet_blocks import SqueezeExcite, InvertedResidual
from .efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights,\
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from .features import FeatureInfo, FeatureHooks
from .helpers import build_model_with_cfg, default_cfg_for_features
from .layers import create_conv2d, create_classifier
from .registry import register_model
sys.path.append('../../')
from tools import global_var

__all__ = ['EfficientNet', 'EfficientNetFeatures']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mnasnet_050': _cfg(url=''),
    'mnasnet_075': _cfg(url=''),
    'mnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth'),
    'mnasnet_140': _cfg(url=''),

    'semnasnet_050': _cfg(url=''),
    'semnasnet_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/semnasnet_075-18710866.pth'),
    'semnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth'),
    'semnasnet_140': _cfg(url=''),
    'mnasnet_small': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_small_lamb-aff75073.pth'),

    'mobilenetv2_035': _cfg(
        url=''),
    'mobilenetv2_050': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_050-3d30d450.pth',
        interpolation='bicubic',
    ),
    'mobilenetv2_075': _cfg(
        url=''),
    'mobilenetv2_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth'),
    'mobilenetv2_110d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_110d_ra-77090ade.pth'),
    'mobilenetv2_120d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_120d_ra-5987e2ed.pth'),
    'mobilenetv2_140': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_140_ra-21a4e913.pth'),

    'fbnetc_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
        interpolation='bilinear'),
    'spnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',
        interpolation='bilinear'),

    # NOTE experimenting with alternate attention
    'efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth'),
    'efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        test_input_size=(3, 256, 256), crop_pct=1.0),
    'efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), test_input_size=(3, 288, 288), crop_pct=1.0),
    'efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), crop_pct=1.0),
    'efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), test_input_size=(3, 384, 384), crop_pct=1.0),
    'efficientnet_b5': _cfg(
        url='', input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'efficientnet_b6': _cfg(
        url='', input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'efficientnet_b7': _cfg(
        url='', input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'efficientnet_b8': _cfg(
        url='', input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
    'efficientnet_l2': _cfg(
        url='', input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.961),

    'efficientnet_es': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth'),
    'efficientnet_em': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_em_ra2-66250f76.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_el': _cfg(
        url='https://github.com/DeGirum/pruned-models/releases/download/efficientnet_v1.0/efficientnet_el.pth', 
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'efficientnet_es_pruned': _cfg(
        url='https://github.com/DeGirum/pruned-models/releases/download/efficientnet_v1.0/efficientnet_es_pruned75.pth'),
    'efficientnet_el_pruned': _cfg(
        url='https://github.com/DeGirum/pruned-models/releases/download/efficientnet_v1.0/efficientnet_el_pruned70.pth', 
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'efficientnet_cc_b0_4e': _cfg(url=''),
    'efficientnet_cc_b0_8e': _cfg(url=''),
    'efficientnet_cc_b1_8e': _cfg(url='', input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),

    'efficientnet_lite0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth'),
    'efficientnet_lite1': _cfg(
        url='',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_lite2': _cfg(
        url='',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'efficientnet_lite3': _cfg(
        url='',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'efficientnet_lite4': _cfg(
        url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),

    'efficientnet_b1_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb1_pruned_9ebb3fe6.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b2_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb2_pruned_203f55bc.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'efficientnet_b3_pruned': _cfg(
        url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45403/outputs/effnetb3_pruned_5abcc29f.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'efficientnetv2_rw_t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth',
        input_size=(3, 224, 224), test_input_size=(3, 288, 288), pool_size=(7, 7), crop_pct=1.0),
    'gc_efficientnetv2_rw_t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gc_efficientnetv2_rw_t_agc-927a0bde.pth',
        input_size=(3, 224, 224), test_input_size=(3, 288, 288), pool_size=(7, 7), crop_pct=1.0),
    'efficientnetv2_rw_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_v2s_ra2_288-a6477665.pth',
        input_size=(3, 288, 288), test_input_size=(3, 384, 384), pool_size=(9, 9), crop_pct=1.0),
    'efficientnetv2_rw_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_rw_m_agc-3d90cb1e.pth',
        input_size=(3, 320, 320), test_input_size=(3, 416, 416), pool_size=(10, 10), crop_pct=1.0),

    'efficientnetv2_s': _cfg(
        url='',
        input_size=(3, 288, 288), test_input_size=(3, 384, 384), pool_size=(9, 9), crop_pct=1.0),
    'efficientnetv2_m': _cfg(
        url='',
        input_size=(3, 320, 320), test_input_size=(3, 416, 416), pool_size=(10, 10), crop_pct=1.0),
    'efficientnetv2_l': _cfg(
        url='',
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'efficientnetv2_xl': _cfg(
        url='',
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth',
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),

    'tf_efficientnet_b0_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ap-f262efe1.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, input_size=(3, 224, 224)),
    'tf_efficientnet_b1_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ap-44ef0a3d.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ap-2f8e7636.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ap-aad25bdd.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ap-dedb23e6.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ap-4ffb161f.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ap-ddb28fec.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ap-00e169fa.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),

    'tf_efficientnet_b0_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ns-c0e6a31c.pth',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ns-99dd0c41.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ns-9d44bf68.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_l2_ns_475': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns_475-bebbd00a.pth',
        input_size=(3, 475, 475), pool_size=(15, 15), crop_pct=0.936),
    'tf_efficientnet_l2_ns': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth',
        input_size=(3, 800, 800), pool_size=(25, 25), crop_pct=0.96),

    'tf_efficientnet_es': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 224, 224), ),
    'tf_efficientnet_em': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_el': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),

    'tf_efficientnet_cc_b0_4e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_4e-4362b6b2.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b0_8e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_8e-66184a25.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b1_8e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b1_8e-f7c79ae1.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),

    'tf_efficientnet_lite0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890,
        interpolation='bicubic',  # should be bilinear but bicubic better match for TF bilinear at low res
    ),
    'tf_efficientnet_lite3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904, interpolation='bilinear'),
    'tf_efficientnet_lite4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.920, interpolation='bilinear'),

    'tf_efficientnetv2_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s-eb54923e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m-cc09e0cd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l-d664b728.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnetv2_s_in21ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s_21ft1k-d7dafa41.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m_in21ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21ft1k-bf41664a.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_l_in21ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21ft1k-60127a9d.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_xl_in21ft1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_xl_in21ft1k-06c35c48.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnetv2_s_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s_21k-6337ad01.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 300, 300), test_input_size=(3, 384, 384), pool_size=(10, 10), crop_pct=1.0),
    'tf_efficientnetv2_m_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21k-361418a2.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_l_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21k-91a19ec9.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 480, 480), pool_size=(12, 12), crop_pct=1.0),
    'tf_efficientnetv2_xl_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_xl_in21k-fd7e8abf.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_classes=21843,
        input_size=(3, 384, 384), test_input_size=(3, 512, 512), pool_size=(12, 12), crop_pct=1.0),

    'tf_efficientnetv2_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth',
        input_size=(3, 192, 192), test_input_size=(3, 224, 224), pool_size=(6, 6)),
    'tf_efficientnetv2_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b1-be6e41b0.pth',
        input_size=(3, 192, 192), test_input_size=(3, 240, 240), pool_size=(6, 6), crop_pct=0.882),
    'tf_efficientnetv2_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b2-847de54e.pth',
        input_size=(3, 208, 208), test_input_size=(3, 260, 260), pool_size=(7, 7), crop_pct=0.890),
    'tf_efficientnetv2_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b3-57773f13.pth',
        input_size=(3, 240, 240), test_input_size=(3, 300, 300), pool_size=(8, 8), crop_pct=0.904),

    'mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth'),
    'mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth'),
    'mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth'),
    'mixnet_xl': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth'),
    'mixnet_xxl': _cfg(),

    'tf_mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth'),
    'tf_mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth'),
    'tf_mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth'),

    "tinynet_a": _cfg(
        input_size=(3, 192, 192), pool_size=(6, 6),  # int(224 * 0.86)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_a.pth'),
    "tinynet_b": _cfg(
        input_size=(3, 188, 188), pool_size=(6, 6),  # int(224 * 0.84)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_b.pth'),
    "tinynet_c": _cfg(
        input_size=(3, 184, 184), pool_size=(6, 6),  # int(224 * 0.825)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_c.pth'),
    "tinynet_d": _cfg(
        input_size=(3, 152, 152), pool_size=(5, 5),  # int(224 * 0.68)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_d.pth'),
    "tinynet_e": _cfg(
        input_size=(3, 106, 106), pool_size=(4, 4),  # int(224 * 0.475)
        url='https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_e.pth'),
}

# class slimInvertedResidual(nn.Module):
#     def __init__(self, inp, outp, cmid, stride):
#         super(slimInvertedResidual, self).__init__()
#         assert stride in [1, 2]
#
#         self.residual_connection = stride == 1 and inp == outp
#
#         layers = []
#         # expand
#         if cmid != inp:
#             layers += [
#                 nn.Conv2d(inp, cmid, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(cmid),
#                 nn.ReLU6(inplace=True),
#             ]
#         # depthwise + project back
#         layers += [
#             nn.Conv2d(
#                 cmid, cmid, 3, stride, 1, groups=cmid, bias=False),
#             nn.BatchNorm2d(cmid),
#             nn.ReLU6(inplace=True),
#             nn.Conv2d(cmid, outp, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(outp),
#         ]
#         self.body = nn.Sequential(*layers)


class EfficientNet(nn.Module):
    """ (Generic) EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * MobileNet-V2
      * FBNet C
      * Single-Path NAS Pixel1

    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False,
                 output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None,
                 se_layer=None, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features)
        self.act2 = act_layer(inplace=True)
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        # self.block_lens = []
        block_len = 0
        for i in range(len(self.blocks)):
            # self.block_lens.append(len(self.blocks[i]))
            block_len += len(self.blocks[i])
        self.block_len = block_len
        self.norm_layer = norm_layer
        self.padtype = pad_type
        self.global_pool_type = global_pool

    def get_pruned_module(self, module_list):
        for blockidx in range(len(module_list)):
            for blockchoice in range(len(module_list[blockidx])):
                self.multiblocks[blockidx+1].append(module_list[blockidx][blockchoice])
        for i in range(1, len(self.block_choices) - 1):
            self.block_choices[i] += [-1, -2]

    def get_pruned_modulev2(self, module_list, prune_checkpoints, prune_block_nums):
        '''v2 means this function adds blocks that are pruned in a set, not just one'''
        self.prune_checkpoints = prune_checkpoints
        self.prune_block_nums = prune_block_nums
        option_num = len(module_list)
        for prune_rate_idx in range(len(module_list)):
            for blockidx in range(len(module_list[prune_rate_idx])):
                self.multiblocks[prune_checkpoints[blockidx]].append(module_list[prune_rate_idx][blockidx])
                self.block_choices[prune_checkpoints[blockidx]].append(prune_rate_idx - option_num)
        # for blockidx in range(len(module_list)):
        #     # for blockchoice in range(len(module_list[blockidx])):
        #         self.multiblocks[prune_checkpoints[blockidx]].append(module_list[blockidx][blockchoice])
                # self.block_choices[prune_checkpoints[blockidx]] += [-2, -1]

    def get_multi_blocks(self, choices=1, block_len=None, norm_layer=None):
        if block_len is None:
            block_len = self.block_len
        if norm_layer is None:
            norm_layer = self.norm_layer
        self.multiblocks = nn.ModuleList()
        if choices == 0:
            for stage_idx in range(len(self.blocks)):
                for block_idx in range(len(self.blocks[stage_idx])):
                    self.multiblocks.append(nn.ModuleList())
                    self.multiblocks[-1].append(self.blocks[stage_idx][block_idx])
            return
        multi_block_idx = 0
        # input_channel = 32
        # self.block_state_dict = []
        self.block_choices = []
        for stage_idx in range(len(self.blocks)):
            '''for efficientnet tiny specially'''
            # if stage_idx == 0:
            #     for block_idx in range(len(self.blocks[stage_idx])):
            #         self.multiblocks.append(nn.ModuleList())
            #         self.multiblocks[-1].append(self.blocks[stage_idx][block_idx])
            #         self.block_choices.append([])
            #         self.block_choices[-1].append(0)
            #         if block_idx == 0:
            #             self.multiblocks[-1].append(self.blocks[stage_idx][block_idx])
            #             self.block_choices[-1].append(1)
            #     multi_block_idx += 2
            #     continue
            '''end'''
            for block_idx in range(len(self.blocks[stage_idx])):
                self.multiblocks.append(nn.ModuleList())
                self.multiblocks[-1].append(self.blocks[stage_idx][block_idx])
                self.block_choices.append([])
                self.block_choices[-1].append(0)
                this_stride, this_inchs, this_outchs = self.get_block_info(self.blocks[stage_idx][block_idx])
                mid_chs = int(this_inchs * 6)
                if hasattr(self.blocks[stage_idx][block_idx], 'mid_chs') and self.blocks[stage_idx][block_idx].mid_chs != mid_chs:
                    mid_chs = self.blocks[stage_idx][block_idx].mid_chs
                if multi_block_idx <= block_len - 2:
                    next_block = self.get_next_block(stage_idx, block_idx)
                    distill_next = copy.deepcopy(self.blocks[stage_idx][block_idx])

                    next_stride, next_inchs, next_outchs = self.get_block_info(next_block)
                    stride = max(next_stride, this_stride)
                    if next_stride + this_stride <= 3:
                        if hasattr(self.blocks[stage_idx][block_idx], 'conv_dw'):
                            if hasattr(self.blocks[stage_idx][block_idx], 'conv_pwl'):  # inverted residual
                                # mid_chs = this_inchs * 6
                                distill_next.conv_pw = nn.Conv2d(this_inchs, mid_chs, kernel_size=1, bias=False)
                                distill_next.bn1 = nn.BatchNorm2d(mid_chs)
                                distill_next.conv_dw = create_conv2d(
                                    # self.blocks[stage_idx][block_idx].mid_chs, self.blocks[stage_idx][block_idx].mid_chs,
                                    mid_chs, mid_chs,
                                    3, stride=stride, dilation=self.blocks[stage_idx][block_idx].dilation,
                                    padding=self.blocks[stage_idx][block_idx].pad_type, depthwise=True,
                                    **self.blocks[stage_idx][block_idx].conv_kwargs)
                                distill_next.bn2 = nn.BatchNorm2d(mid_chs)
                                distill_next.has_residual = (this_inchs == next_outchs and stride == 1)
                                distill_next.conv_pwl = create_conv2d(mid_chs, #self.blocks[stage_idx][block_idx].mid_chs,
                                                                      next_outchs, 1, padding=self.blocks[stage_idx][block_idx].pad_type,
                                                                      **self.blocks[stage_idx][block_idx].conv_kwargs)
                                distill_next.bn3 = norm_layer(next_outchs)
                            else:  # DepthwiseSeparableConv
                                distill_next.conv_dw = create_conv2d(this_inchs, self.blocks[stage_idx][block_idx].conv_pw.in_channels, 3, stride=stride,
                                                                     padding=self.blocks[stage_idx][block_idx].pad_type, depthwise=True)
                                distill_next.conv_pw = create_conv2d(self.blocks[stage_idx][block_idx].conv_pw.in_channels, next_outchs, 1,
                                    stride=1, padding=self.blocks[stage_idx][block_idx].pad_type, depthwise=False)
                                distill_next.has_residual = (this_inchs == next_outchs and stride == 1)
                                distill_next.bn2 = norm_layer(next_outchs)

                        else:  # edge residual
                            distill_next.conv_exp = create_conv2d(
                                this_inchs, self.blocks[stage_idx][block_idx].mid_chs, 3, stride=stride,
                                dilation=self.blocks[stage_idx][block_idx].dilation, padding=self.blocks[stage_idx][block_idx].pad_type)
                            distill_next.has_residual = (this_inchs == next_outchs and stride == 1) and not self.blocks[stage_idx][block_idx].noskip
                            distill_next.conv_pwl = create_conv2d(self.blocks[stage_idx][block_idx].mid_chs, next_outchs,
                                                                  1, padding=self.blocks[stage_idx][block_idx].pad_type)
                            distill_next.bn2 = norm_layer(next_outchs)

                        self.multiblocks[-1].append(distill_next)
                        self.block_choices[-1].append(1)

                if multi_block_idx <= block_len - 3:
                    # print(stage_idx, block_idx)
                    next_next_block = self.get_next_next_block(stage_idx, block_idx)
                    distill_next_next = copy.deepcopy(self.blocks[stage_idx][block_idx])
                    # next_stride, next_inchs, next_outchs = self.get_block_info(next_block)

                    next_next_stride, next_next_inchs, next_next_outchs = self.get_block_info(next_next_block)
                    if this_stride + next_stride + next_next_stride <= 4:
                        stride = max(this_stride, next_stride, next_next_stride)

                        # if mid_chs <
                        if hasattr(self.blocks[stage_idx][block_idx], 'conv_dw'):
                            if hasattr(self.blocks[stage_idx][block_idx], 'conv_pwl'):  # inverted residual
                                # mid_chs = this_inchs * 6
                                distill_next_next.conv_pw = nn.Conv2d(this_inchs, mid_chs, kernel_size=1, bias=False)
                                distill_next_next.bn1 = nn.BatchNorm2d(mid_chs)
                                distill_next_next.conv_dw = create_conv2d(
                                    # self.blocks[stage_idx][block_idx].mid_chs, self.blocks[stage_idx][block_idx].mid_chs,
                                    mid_chs, mid_chs,
                                    3, stride=stride, dilation=self.blocks[stage_idx][block_idx].dilation,
                                    padding=self.blocks[stage_idx][block_idx].pad_type, depthwise=True,
                                    **self.blocks[stage_idx][block_idx].conv_kwargs)
                                distill_next_next.bn2 = nn.BatchNorm2d(mid_chs)
                                distill_next_next.has_residual = (this_inchs == next_next_outchs and stride == 1)
                                distill_next_next.conv_pwl = create_conv2d(mid_chs, #self.blocks[stage_idx][block_idx].mid_chs,
                                                                           next_next_outchs, 1, padding=self.blocks[stage_idx][block_idx].pad_type,
                                                                      **self.blocks[stage_idx][block_idx].conv_kwargs)
                                distill_next_next.bn3 = norm_layer(next_next_outchs)
                            else:  # DepthwiseSeparableConv
                                distill_next_next.conv_dw = create_conv2d(this_inchs, self.blocks[stage_idx][block_idx].conv_pw.in_channels, 3, stride=stride,
                                                                     padding=self.blocks[stage_idx][block_idx].pad_type, depthwise=True)
                                distill_next_next.conv_pw = create_conv2d(self.blocks[stage_idx][block_idx].conv_pw.in_channels, next_next_outchs, 1, stride=1,
                                                                     padding=self.blocks[stage_idx][block_idx].pad_type, depthwise=False)
                                distill_next_next.has_residual = (this_inchs == next_next_outchs and stride == 1)
                                distill_next_next.bn2 = norm_layer(next_next_outchs)
                        else:  # edge residual
                            distill_next_next.conv_exp = create_conv2d(
                                this_inchs, self.blocks[stage_idx][block_idx].mid_chs, 3, stride=stride,
                                dilation=self.blocks[stage_idx][block_idx].dilation, padding=self.blocks[stage_idx][block_idx].pad_type)
                            distill_next_next.has_residual = (this_inchs == next_next_outchs and stride == 1) and not self.blocks[stage_idx][block_idx].noskip
                            distill_next_next.conv_pwl = create_conv2d(self.blocks[stage_idx][block_idx].mid_chs, next_next_outchs,
                                                                  1, padding=self.blocks[stage_idx][block_idx].pad_type)
                            distill_next_next.bn2 = norm_layer(next_next_outchs)
                        self.multiblocks[-1].append(distill_next_next)
                        self.block_choices[-1].append(2)
                multi_block_idx += 1
                    # state_dict[1] = 1
                    # temp += 1
        del self.blocks
        efficientnet_init_weights(self)

    def get_next_block(self, stage_idx, block_idx):
        if block_idx < (len(self.blocks[stage_idx]) - 1):
            # if stage_idx < 3:
            #     block_type = "EdgeResidual"
            # else:
            #     block_type = "InvertedResidual"
            return self.blocks[stage_idx][block_idx + 1]#, block_type
        elif stage_idx < (len(self.blocks) - 1):
            # if (stage_idx + 1) < 3:
            #     block_type = "EdgeResidual"
            # else:
            #     block_type = "InvertedResidual"
            return self.blocks[stage_idx + 1][0]#, block_type

    def adjust_inverted_residual(self, ch_in, ch_mid, ch_out, stride):
        block = InvertedResidual(ch_in, ch_out, stride=stride, exp_ratio=ch_mid/ch_in)
        return block

    def adjust_channels(self, channels=None, strides=None):
        if channels is None:
            channels = [32, 16, 144, 24, 176, 24, 192, 48, 240, 48, 144, 48, 264, 88, 288, 88, 336, 88, 432, 88, 576,
                        144, 576, 144, 648, 144, 864, 240, 1440, 240, 1440, 240, 1440, 480, 1920, 1000]
        if strides is None:
            strides = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
        blockidx = 1
        for layeridx in range(1, len(self.blocks)):
            for invertedresidual_idx in range(len(self.blocks[layeridx])):
                ch_in = channels[2*blockidx-1]
                ch_mid = channels[2*blockidx]
                ch_out = channels[2*blockidx+1]
                stride = strides[blockidx]
                self.blocks[layeridx][invertedresidual_idx] = self.adjust_inverted_residual(ch_in, ch_mid, ch_out, stride)
                blockidx += 1
        self.conv_head = create_conv2d(480, 1920, 1, padding=self.padtype)
        self.bn2 = self.norm_layer(1920)
        self.global_pool, self.classifier = create_classifier(
            1920, self.num_classes, pool_type=self.global_pool_type)
        # self.global_pool = torch.nn.AvgPool2d(kernel_size=7, stride=7, padding=0)



    def get_next_next_block(self, stage_idx, block_idx):
        if block_idx < (len(self.blocks[stage_idx]) - 2):
            return self.blocks[stage_idx][block_idx + 2]
        elif (block_idx == (len(self.blocks[stage_idx]) - 2)) and (stage_idx < (len(self.blocks) - 1)):
            return self.blocks[stage_idx + 1][0]
        elif (block_idx == (len(self.blocks[stage_idx]) - 1)) and (stage_idx < (len(self.blocks) - 1)):
            return self.blocks[stage_idx + 1][1]


    def get_block_info(self, block):
        if hasattr(block, 'conv_exp'):  # edgeresidual
            return block.conv_exp.stride[0], block.conv_exp.in_channels, block.conv_pwl.out_channels
        elif hasattr(block, 'conv_dw'):
            if hasattr(block, 'conv_pwl'): # invertedresidual
                return block.conv_dw.stride[0], block.conv_pw.in_channels, block.conv_pwl.out_channels
            else:  # DepthwiseSeparableConv
                return 1, block.conv_dw.in_channels, block.conv_pw.out_channels

    # def get_block_output(self, block):

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def get_next_pruned_checkpoint(self, checkpoint):
        i = 0
        while i < len(self.prune_checkpoints):
            if self.prune_checkpoints[i] == checkpoint:
                break
            i += 1
        return checkpoint + self.prune_block_nums[i], self.prune_block_nums[i]

    def generate_random_subnet(self):
        blockidx = 0
        subnet = []
        # for blockidx in range(self.block_len):
        while blockidx < self.block_len:
            # choices = [0]  # origin block
            # if blockidx < (self.block_len - 1):
            #     choices.append(1)  # distill next one
            # if blockidx < (self.block_len - 2):
            #     choices.append(2)  # distill next two
            if blockidx == 0:
                subnet.append(0)
                blockidx += 1
            else:
                choices = self.block_choices[blockidx]
                choice = np.random.choice(choices)
                if choice == 1:
                    subnet += [1, 99]
                    blockidx += 2
                elif choice == 2:
                    subnet += [2, 99, 99]
                    blockidx += 3
                else:
                    subnet.append(choice)
                    blockidx += 1
                # elif choice == 0:
                #     subnet.append(0)
                #     blockidx += 1
                # else:  # pruning blocks choices
                #     subnet.append(choice)
                #     idx, prune_block_num = self.get_next_pruned_checkpoint(blockidx)
                #     subnet += [99 for _ in range(prune_block_num - 1)]
                #     blockidx = idx
        return subnet

    def forward_all_module(self, x, subnet):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        blockidx = 0
        while blockidx < len(subnet):
            x = self.multiblocks[blockidx][subnet[blockidx]](x)
            if subnet[blockidx] == 1:
                blockidx += 2
            elif subnet[blockidx] == 2:
                blockidx += 3
            else:
                blockidx += 1
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def check_validated_feature(self):
        self.net = global_var.get_value('subnet')
        self.subnet_pre = global_var.get_value('subnet_pre') #本次推理需要保存的前缀
        self.subnet_shared_block = global_var.get_value('subnet_shared_block') #本次推理可以节省的部分
        self.saved_feature = global_var.get_value('inter_features')
        # for k,v in self.saved_feature.items():
        #     print(k)
        if self.subnet_pre == self.subnet_shared_block:
            self.subnet_pre = '-1'
        # print()
        # print('check_validated_feature')
        # print('net:{}'.format(self.net))
        # print('subnet_pre {}'.format(self.subnet_pre))
        # print('subnet_shared_block {}'.format(self.subnet_shared_block))
        if self.saved_feature.__contains__(self.subnet_shared_block):
            self.saved_feature = self.saved_feature[self.subnet_shared_block] #内存中存放的中间特征值.
            # for k,v in self.saved_feature.items():
            #     print(k)

        if self.subnet_shared_block == '-1':
            self.subnet_shared_index = -1
        else:
            self.subnet_shared_index = len(self.subnet_shared_block)

        if self.subnet_pre == '-1':
            self.subnet_pre_block_size = -1
        else:
            self.subnet_pre_block_size = len(self.subnet_pre)


        self.inter_feature = dict()

        self.flag = 1
        self.to_gpu_time = 0
        self.to_cpu_time = 0
        # t1 = time.time()
        # for i in range(len(self.net)):
        #         if self.net[i] == 1 or self.net[i] == 2:
        #             self.multiblocks[i][self.net[i]].cuda()
        # self.to_gpu_time = time.time()-t1
    
    def forward_baseline(self, x, subnet):
        blockidx = 1
        while blockidx < len(subnet):
            x = self.multiblocks[blockidx][subnet[blockidx]](x)
            if subnet[blockidx] == 1:
                blockidx += 2
            elif subnet[blockidx] == 2:
                blockidx += 3
            else:
                blockidx += 1
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x 
    
    def update_validated_feature(self):
        if self.subnet_pre != -1:
            #print('update_validated_feature:{}'.format(self.subnet_pre))
            inter_features = global_var.get_value('inter_features')
            inter_features[self.subnet_pre] = self.inter_feature
                
            global_var.set_value('inter_features',inter_features)
            # t1 = time.time()
            # for i in range(len(self.net)):
            #     if self.net[i] == 1 or self.net[i] == 2:
            #         self.multiblocks[i][self.net[i]].cuda()
            # self.to_cpu_time = time.time()-t1

            # print('to_cpu_time:{}'.format(self.to_cpu_time))
            # print('to_gpu_time:{}'.format(self.to_gpu_time))

    def forward_features(self, x, subnet,batch_idx=-1):
        block_len = len(self.multiblocks)
        if self.subnet_shared_index == -1:
            blockidx = 1
            while blockidx < len(subnet):
                if self.subnet_pre_block_size == blockidx:
                    subnet_pre_block_name = '{}-{}.pt'.format(batch_idx,self.subnet_pre)
                    self.inter_feature[subnet_pre_block_name] = x
                x = self.multiblocks[blockidx][subnet[blockidx]](x)
                if subnet[blockidx] == 1:
                    blockidx += 2
                elif subnet[blockidx] == 2:
                    blockidx += 3
                else:
                    blockidx += 1
            if self.subnet_pre_block_size == blockidx:
                    subnet_pre_block_name = '{}-{}.pt'.format(batch_idx,self.subnet_pre)
                    self.inter_feature[subnet_pre_block_name] = x
        else:
            block_ind  = 0
            subnet_shared_block_name =  '{}-{}.pt'.format(batch_idx,self.subnet_shared_block)
            x = self.saved_feature[subnet_shared_block_name]
            blockidx = self.subnet_shared_index
            while blockidx < len(subnet):
                if self.subnet_pre_block_size == blockidx:
                    subnet_name = '{}-{}.pt'.format(batch_idx,self.subnet_pre)
                    self.inter_feature[subnet_name] = x
                #print(blockidx,len(subnet),subnet,x.shape)
                x = self.multiblocks[blockidx][subnet[blockidx]](x)
                if subnet[blockidx] == 1:
                    blockidx += 2
                elif subnet[blockidx] == 2:
                    blockidx += 3
                else:
                    blockidx += 1
            if self.subnet_pre_block_size == blockidx:
                    subnet_pre_block_name = '{}-{}.pt'.format(batch_idx,self.subnet_pre)
                    self.inter_feature[subnet_pre_block_name] = x
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
    
    def forward_normal(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        blockidx = 0
        while blockidx < len(self.subnet):
            x = self.multiblocks[blockidx][self.subnet[blockidx]](x)
            if self.subnet[blockidx] == 1:
                blockidx += 2
            elif self.subnet[blockidx] == 2:
                blockidx += 3
            else:
                blockidx += 1
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
    
    def apply_subnet(self, subnet):
        self.subnet = subnet

    def forward(self, x, subnet=None,infer_type=2,batch_idx=-1):
        if subnet is not None:
            if infer_type == 0:
                x = self.forward_all_module(x, subnet)
            elif infer_type == 1:
                x = self.forward_baseline(x, subnet)
            else:
                x = self.forward_features(x,subnet,batch_idx)
        else:
            x = self.forward_normal(x)

        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)

class EfficientNetFeatures(nn.Module):
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3,
                 stem_size=32, fix_stem=False, output_stride=32, pad_type='', round_chs_fn=round_channels,
                 act_layer=None, norm_layer=None, se_layer=None, drop_rate=0., drop_path_rate=0.):
        super(EfficientNetFeatures, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate,
            feature_location=feature_location)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}

        efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def _create_effnet(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = EfficientNet
    kwargs_filter = None
    # if kwargs.pop('features_only', False):
    #     features_only = True
    #     kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'global_pool')
    #     model_cls = EfficientNetFeatures
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    if features_only:
        model.default_cfg = default_cfg_for_features(model.default_cfg)
    return model


def _gen_mnasnet_a1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-a1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r2_k3_s2_e6_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r4_k3_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mnasnet_b1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r3_k5_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mnasnet_small(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        ['ds_r1_k3_s1_c8'],
        ['ir_r1_k3_s2_e3_c16'],
        ['ir_r2_k3_s2_e6_c16'],
        ['ir_r4_k5_s2_e6_c32_se0.25'],
        ['ir_r3_k3_s1_e6_c32_se0.25'],
        ['ir_r3_k5_s2_e6_c88_se0.25'],
        ['ir_r1_k3_s1_e6_c144']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=8,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_fbnetc(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNet-C

        Paper: https://arxiv.org/abs/1812.03443
        Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

        NOTE: the impl above does not relate to the 'C' variant here, that was derived from paper,
        it was used to confirm some building block details
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c16'],
        ['ir_r1_k3_s2_e6_c24', 'ir_r2_k3_s1_e1_c24'],
        ['ir_r1_k5_s2_e6_c32', 'ir_r1_k5_s1_e3_c32', 'ir_r1_k5_s1_e6_c32', 'ir_r1_k3_s1_e6_c32'],
        ['ir_r1_k5_s2_e6_c64', 'ir_r1_k5_s1_e3_c64', 'ir_r2_k5_s1_e6_c64'],
        ['ir_r3_k5_s1_e6_c112', 'ir_r1_k5_s1_e3_c112'],
        ['ir_r4_k5_s2_e6_c184'],
        ['ir_r1_k3_s1_e6_c352'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        num_features=1984,  # paper suggests this, but is not 100% clear
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_spnasnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates the Single-Path NAS model from search targeted for Pixel1 phone.

    Paper: https://arxiv.org/abs/1904.02877

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e6_c40', 'ir_r3_k3_s1_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r1_k5_s2_e6_c80', 'ir_r3_k3_s1_e3_c80'],
        # stage 4, 14x14in
        ['ir_r1_k5_s1_e6_c96', 'ir_r3_k5_s1_e3_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet_edge(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-EdgeTPU model

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu
    """

    arch_def = [
        # NOTE `fc` is present to override a mismatch between stem channels and in chs not
        # present in other models
        ['er_r1_k3_s1_e4_c24_fc24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet_condconv(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=1, pretrained=False, **kwargs):
    """Creates an EfficientNet-CondConv model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25_cc4'],
        ['ir_r4_k5_s2_e6_c192_se0.25_cc4'],
        ['ir_r1_k3_s1_e6_c320_se0.25_cc4'],
    ]
    # NOTE unlike official impl, this one uses `cc<x>` option where x is the base number of experts for each stage and
    # the expert_multiplier increases that on a per-model basis as with depth/channel multipliers
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, experts_multiplier=experts_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'swish'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet_lite(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_base(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 base model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """
    arch_def = [
        ['cn_r1_k3_s1_e1_c16_skip'],
        ['er_r2_k3_s2_e4_c32'],
        ['er_r2_k3_s2_e4_c48'],
        ['ir_r3_k3_s2_e4_c96_se0.25'],
        ['ir_r5_k3_s1_e6_c112_se0.25'],
        ['ir_r8_k3_s2_e6_c192_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_s(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, rw=False, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Small model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298

    NOTE: `rw` flag sets up 'small' variant to behave like my initial v2 small model,
        before ref the impl was released.
    """
    arch_def = [
        ['cn_r2_k3_s1_e1_c24_skip'],
        ['er_r4_k3_s2_e4_c48'],
        ['er_r4_k3_s2_e4_c64'],
        ['ir_r6_k3_s2_e4_c128_se0.25'],
        ['ir_r9_k3_s1_e6_c160_se0.25'],
        ['ir_r15_k3_s2_e6_c256_se0.25'],
    ]
    num_features = 1280
    if rw:
        # my original variant, based on paper figure differs from the official release
        arch_def[0] = ['er_r2_k3_s1_e1_c24']
        arch_def[-1] = ['ir_r15_k3_s2_e6_c272_se0.25']
        num_features = 1792

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(num_features),
        stem_size=24,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Medium model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_l(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Large model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnetv2_xl(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """ Creates an EfficientNet-V2 Xtra-Large model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    """

    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r8_k3_s2_e4_c64'],
        ['er_r8_k3_s2_e4_c96'],
        ['ir_r16_k3_s2_e4_c192_se0.25'],
        ['ir_r24_k3_s1_e6_c256_se0.25'],
        ['ir_r32_k3_s2_e6_c512_se0.25'],
        ['ir_r8_k3_s1_e6_c640_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mixnet_s(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Small model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_a1.1_p1.1_s2_e6_c24', 'ir_r1_k3_a1.1_p1.1_s1_e3_c24'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_p1.1_s2_e6_c80_se0.25_nsw', 'ir_r2_k3.5_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3.5.7_a1.1_p1.1_s1_e6_c120_se0.5_nsw', 'ir_r2_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9.11_s2_e6_c200_se0.5_nsw', 'ir_r2_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1536,
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mixnet_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Medium-Large model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c24'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c32', 'ir_r1_k3_a1.1_p1.1_s1_e3_c32'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7.9_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_s2_e6_c80_se0.25_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c120_se0.5_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9_s2_e6_c200_se0.5_nsw', 'ir_r3_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
        num_features=1536,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_tinynet(
    variant, model_width=1.0, depth_multiplier=1.0, pretrained=False, **kwargs
):
    """Creates a TinyNet model.
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
        num_features=max(1280, round_channels(1280, model_width, 8, None)),
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=model_width),
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


@register_model
def mnasnet_050(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.5. """
    model = _gen_mnasnet_b1('mnasnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_075(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.75. """
    model = _gen_mnasnet_b1('mnasnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_100(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    model = _gen_mnasnet_b1('mnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_b1(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    return mnasnet_100(pretrained, **kwargs)


@register_model
def mnasnet_140(pretrained=False, **kwargs):
    """ MNASNet B1,  depth multiplier of 1.4 """
    model = _gen_mnasnet_b1('mnasnet_140', 1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def semnasnet_050(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 0.5 """
    model = _gen_mnasnet_a1('semnasnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def semnasnet_075(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE),  depth multiplier of 0.75. """
    model = _gen_mnasnet_a1('semnasnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def semnasnet_100(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    model = _gen_mnasnet_a1('semnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_a1(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    return semnasnet_100(pretrained, **kwargs)


@register_model
def semnasnet_140(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.4. """
    model = _gen_mnasnet_a1('semnasnet_140', 1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def mnasnet_small(pretrained=False, **kwargs):
    """ MNASNet Small,  depth multiplier of 1.0. """
    model = _gen_mnasnet_small('mnasnet_small', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_035(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 0.35 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_050(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 0.5 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_075(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 0.75 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_100(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.0 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_140(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.4 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_140', 1.4, pretrained=pretrained, **kwargs)
    return model

@register_model
def mobilenetv2_150(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.4 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_150', 1.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_110d(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.1 channel, 1.2 depth multipliers"""
    model = _gen_mobilenet_v2(
        'mobilenetv2_110d', 1.1, depth_multiplier=1.2, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv2_120d(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.2 channel, 1.4 depth multipliers """
    model = _gen_mobilenet_v2(
        'mobilenetv2_120d', 1.2, depth_multiplier=1.4, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetc_100(pretrained=False, **kwargs):
    """ FBNet-C """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_fbnetc('fbnetc_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def spnasnet_100(pretrained=False, **kwargs):
    """ Single-Path NAS Pixel1"""
    model = _gen_spnasnet('spnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b1(pretrained=False, **kwargs):
    """ EfficientNet-B1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2(pretrained=False, **kwargs):
    """ EfficientNet-B2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2a(pretrained=False, **kwargs):
    """ EfficientNet-B2 @ 288x288 w/ 1.0 test crop"""
    # WARN this model def is deprecated, different train/test res + test crop handled by default_cfg now
    return efficientnet_b2(pretrained=pretrained, **kwargs)


@register_model
def efficientnet_b3(pretrained=False, **kwargs):
    """ EfficientNet-B3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b3a(pretrained=False, **kwargs):
    """ EfficientNet-B3 @ 320x320 w/ 1.0 test crop-pct """
    # WARN this model def is deprecated, different train/test res + test crop handled by default_cfg now
    return efficientnet_b3(pretrained=pretrained, **kwargs)


@register_model
def efficientnet_b4(pretrained=False, **kwargs):
    """ EfficientNet-B4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b5(pretrained=False, **kwargs):
    """ EfficientNet-B5 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b6(pretrained=False, **kwargs):
    """ EfficientNet-B6 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b7(pretrained=False, **kwargs):
    """ EfficientNet-B7 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b8(pretrained=False, **kwargs):
    """ EfficientNet-B8 """
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_l2(pretrained=False, **kwargs):
    """ EfficientNet-L2."""
    # NOTE for train, drop_rate should be 0.5, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'efficientnet_l2', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_es(pretrained=False, **kwargs):
    """ EfficientNet-Edge Small. """
    model = _gen_efficientnet_edge(
        'efficientnet_es', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_es_pruned(pretrained=False, **kwargs):
    """ EfficientNet-Edge Small Pruned. For more info: https://github.com/DeGirum/pruned-models/releases/tag/efficientnet_v1.0"""
    model = _gen_efficientnet_edge(
        'efficientnet_es_pruned', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_em(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Medium. """
    model = _gen_efficientnet_edge(
        'efficientnet_em', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_el(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Large. """
    model = _gen_efficientnet_edge(
        'efficientnet_el', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_el_pruned(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Large pruned. For more info: https://github.com/DeGirum/pruned-models/releases/tag/efficientnet_v1.0"""
    model = _gen_efficientnet_edge(
        'efficientnet_el_pruned', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model

@register_model
def efficientnet_cc_b0_4e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 8 Experts """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b0_4e', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_cc_b0_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 8 Experts """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b0_8e', channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_cc_b1_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B1 w/ 8 Experts """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b1_8e', channel_multiplier=1.0, depth_multiplier=1.1, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite0(pretrained=False, **kwargs):
    """ EfficientNet-Lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite1(pretrained=False, **kwargs):
    """ EfficientNet-Lite1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite2(pretrained=False, **kwargs):
    """ EfficientNet-Lite2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite3(pretrained=False, **kwargs):
    """ EfficientNet-Lite3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_lite4(pretrained=False, **kwargs):
    """ EfficientNet-Lite4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    model = _gen_efficientnet_lite(
        'efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b1_pruned(pretrained=False, **kwargs):
    """ EfficientNet-B1 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    variant = 'efficientnet_b1_pruned'
    model = _gen_efficientnet(
        variant, channel_multiplier=1.0, depth_multiplier=1.1, pruned=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b2_pruned(pretrained=False, **kwargs):
    """ EfficientNet-B2 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'efficientnet_b2_pruned', channel_multiplier=1.1, depth_multiplier=1.2, pruned=True,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnet_b3_pruned(pretrained=False, **kwargs):
    """ EfficientNet-B3 Pruned. The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'efficientnet_b3_pruned', channel_multiplier=1.2, depth_multiplier=1.4, pruned=True,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_rw_t(pretrained=False, **kwargs):
    """ EfficientNet-V2 Tiny (Custom variant, tiny not in paper). """
    model = _gen_efficientnetv2_s(
        'efficientnetv2_rw_t', channel_multiplier=0.8, depth_multiplier=0.9, rw=False, pretrained=pretrained, **kwargs)
    return model


@register_model
def gc_efficientnetv2_rw_t(pretrained=False, **kwargs):
    """ EfficientNet-V2 Tiny w/ Global Context Attn (Custom variant, tiny not in paper). """
    model = _gen_efficientnetv2_s(
        'gc_efficientnetv2_rw_t', channel_multiplier=0.8, depth_multiplier=0.9,
        rw=False, se_layer='gc', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_rw_s(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small (RW variant).
    NOTE: This is my initial (pre official code release) w/ some differences.
    See efficientnetv2_s and tf_efficientnetv2_s for versions that match the official w/ PyTorch vs TF padding
    """
    model = _gen_efficientnetv2_s('efficientnetv2_rw_s', rw=True, pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_rw_m(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium (RW variant).
    """
    model = _gen_efficientnetv2_s(
        'efficientnetv2_rw_m', channel_multiplier=1.2, depth_multiplier=(1.2,) * 4 + (1.6,) * 2, rw=True,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_s(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small. """
    model = _gen_efficientnetv2_s('efficientnetv2_s', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_m(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium. """
    model = _gen_efficientnetv2_m('efficientnetv2_m', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_l(pretrained=False, **kwargs):
    """ EfficientNet-V2 Large. """
    model = _gen_efficientnetv2_l('efficientnetv2_l', pretrained=pretrained, **kwargs)
    return model


@register_model
def efficientnetv2_xl(pretrained=False, **kwargs):
    """ EfficientNet-V2 Xtra-Large. """
    model = _gen_efficientnetv2_xl('efficientnetv2_xl', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b1(pretrained=False, **kwargs):
    """ EfficientNet-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b2(pretrained=False, **kwargs):
    """ EfficientNet-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b3(pretrained=False, **kwargs):
    """ EfficientNet-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b4(pretrained=False, **kwargs):
    """ EfficientNet-B4. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b5(pretrained=False, **kwargs):
    """ EfficientNet-B5. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b6(pretrained=False, **kwargs):
    """ EfficientNet-B6. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b7(pretrained=False, **kwargs):
    """ EfficientNet-B7. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b8(pretrained=False, **kwargs):
    """ EfficientNet-B8. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b0_ap(pretrained=False, **kwargs):
    """ EfficientNet-B0 AdvProp. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0_ap', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b1_ap(pretrained=False, **kwargs):
    """ EfficientNet-B1 AdvProp. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1_ap', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b2_ap(pretrained=False, **kwargs):
    """ EfficientNet-B2 AdvProp. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2_ap', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b3_ap(pretrained=False, **kwargs):
    """ EfficientNet-B3 AdvProp. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3_ap', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b4_ap(pretrained=False, **kwargs):
    """ EfficientNet-B4 AdvProp. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4_ap', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b5_ap(pretrained=False, **kwargs):
    """ EfficientNet-B5 AdvProp. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5_ap', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b6_ap(pretrained=False, **kwargs):
    """ EfficientNet-B6 AdvProp. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6_ap', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b7_ap(pretrained=False, **kwargs):
    """ EfficientNet-B7 AdvProp. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7_ap', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b8_ap(pretrained=False, **kwargs):
    """ EfficientNet-B8 AdvProp. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b8_ap', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b0_ns(pretrained=False, **kwargs):
    """ EfficientNet-B0 NoisyStudent. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0_ns', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b1_ns(pretrained=False, **kwargs):
    """ EfficientNet-B1 NoisyStudent. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1_ns', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b2_ns(pretrained=False, **kwargs):
    """ EfficientNet-B2 NoisyStudent. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2_ns', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b3_ns(pretrained=False, **kwargs):
    """ EfficientNet-B3 NoisyStudent. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3_ns', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b4_ns(pretrained=False, **kwargs):
    """ EfficientNet-B4 NoisyStudent. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4_ns', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b5_ns(pretrained=False, **kwargs):
    """ EfficientNet-B5 NoisyStudent. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5_ns', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b6_ns(pretrained=False, **kwargs):
    """ EfficientNet-B6 NoisyStudent. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6_ns', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_b7_ns(pretrained=False, **kwargs):
    """ EfficientNet-B7 NoisyStudent. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7_ns', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_l2_ns_475(pretrained=False, **kwargs):
    """ EfficientNet-L2 NoisyStudent @ 475x475. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_l2_ns_475', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_l2_ns(pretrained=False, **kwargs):
    """ EfficientNet-L2 NoisyStudent. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_l2_ns', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_es(pretrained=False, **kwargs):
    """ EfficientNet-Edge Small. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_es', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_em(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Medium. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_em', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_el(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Large. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_el', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_cc_b0_4e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 4 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_4e', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_cc_b0_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 8 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_8e', channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_cc_b1_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B1 w/ 8 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b1_8e', channel_multiplier=1.0, depth_multiplier=1.1, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite0(pretrained=False, **kwargs):
    """ EfficientNet-Lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite1(pretrained=False, **kwargs):
    """ EfficientNet-Lite1 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite2(pretrained=False, **kwargs):
    """ EfficientNet-Lite2 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite3(pretrained=False, **kwargs):
    """ EfficientNet-Lite3 """
    # NOTE for train, drop_rate should be 0.3, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnet_lite4(pretrained=False, **kwargs):
    """ EfficientNet-Lite4 """
    # NOTE for train, drop_rate should be 0.4, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model



@register_model
def tf_efficientnetv2_s(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_s('tf_efficientnetv2_s', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_m(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_m('tf_efficientnetv2_m', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_l(pretrained=False, **kwargs):
    """ EfficientNet-V2 Large. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_l('tf_efficientnetv2_l', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_s_in21ft1k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_s('tf_efficientnetv2_s_in21ft1k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_m_in21ft1k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_m('tf_efficientnetv2_m_in21ft1k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_l_in21ft1k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Large. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_l('tf_efficientnetv2_l_in21ft1k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_xl_in21ft1k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Xtra-Large. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_xl('tf_efficientnetv2_xl_in21ft1k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_s_in21k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small w/ ImageNet-21k pretrained weights. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_s('tf_efficientnetv2_s_in21k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_m_in21k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium w/ ImageNet-21k pretrained weights. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_m('tf_efficientnetv2_m_in21k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_l_in21k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Large w/ ImageNet-21k pretrained weights. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_l('tf_efficientnetv2_l_in21k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_xl_in21k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Xtra-Large w/ ImageNet-21k pretrained weights. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_xl('tf_efficientnetv2_xl_in21k', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b0(pretrained=False, **kwargs):
    """ EfficientNet-V2-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base('tf_efficientnetv2_b0', pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b1(pretrained=False, **kwargs):
    """ EfficientNet-V2-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b2(pretrained=False, **kwargs):
    """ EfficientNet-V2-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_efficientnetv2_b3(pretrained=False, **kwargs):
    """ EfficientNet-V2-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnetv2_base(
        'tf_efficientnetv2_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_s(pretrained=False, **kwargs):
    """Creates a MixNet Small model.
    """
    model = _gen_mixnet_s(
        'mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_m(pretrained=False, **kwargs):
    """Creates a MixNet Medium model.
    """
    model = _gen_mixnet_m(
        'mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_l(pretrained=False, **kwargs):
    """Creates a MixNet Large model.
    """
    model = _gen_mixnet_m(
        'mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_xl(pretrained=False, **kwargs):
    """Creates a MixNet Extra-Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    model = _gen_mixnet_m(
        'mixnet_xl', channel_multiplier=1.6, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def mixnet_xxl(pretrained=False, **kwargs):
    """Creates a MixNet Double Extra Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    model = _gen_mixnet_m(
        'mixnet_xxl', channel_multiplier=2.4, depth_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mixnet_s(pretrained=False, **kwargs):
    """Creates a MixNet Small model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_s(
        'tf_mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mixnet_m(pretrained=False, **kwargs):
    """Creates a MixNet Medium model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        'tf_mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mixnet_l(pretrained=False, **kwargs):
    """Creates a MixNet Large model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        'tf_mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_a(pretrained=False, **kwargs):
    model = _gen_tinynet('tinynet_a', 1.0, 1.2, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_b(pretrained=False, **kwargs):
    model = _gen_tinynet('tinynet_b', 0.75, 1.1, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_c(pretrained=False, **kwargs):
    model = _gen_tinynet('tinynet_c', 0.54, 0.85, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_d(pretrained=False, **kwargs):
    model = _gen_tinynet('tinynet_d', 0.54, 0.695, pretrained=pretrained, **kwargs)
    return model


@register_model
def tinynet_e(pretrained=False, **kwargs):
    model = _gen_tinynet('tinynet_e', 0.51, 0.6, pretrained=pretrained, **kwargs)
    return model
