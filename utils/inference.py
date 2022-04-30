import os
import json
import argparse

import pandas as pd
import numpy as np

from mmcv import Config
from mmseg.datasets import (build_dataloader, build_dataset)
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel


####################################################################################
CONFIG_DIR = '/opt/ml/input/mmsegmentation/_custom_configs_/deeplabv3_r101-d8/model.py'
WORK_DIR = '/opt/ml/input/mmsegmentation/work_dirs/deeplabv3_r101-d8'
IMG_ROOT = '/opt/ml/input/mmseg/'
EPOCH = 'latest'
SUBMISSION_NAME = '20220430_deeplabv3_r101-d8'
####################################################################################

TEST_JSON = '/opt/ml/input/data/test.json'
SUBMISSION_SAMPLE = '/opt/ml/input/code/submission/sample_submission.csv'
classes = ("Background", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


def parse_args():
    parser = argparse.ArgumentParser(description='Mmsegmentation inference file')

    parser.add_argument('--config_dir', default=CONFIG_DIR, type=str, help='the path of model config file')
    parser.add_argument('--work_dir', default=WORK_DIR, type=str, help='the path of work_dirs in mmsegmentation')
    parser.add_argument('--img_root', default=IMG_ROOT, type=str, help='the root path of images')
    parser.add_argument('--epoch', default='latest', type=str, help='the name of pth file without extension')
    parser.add_argument('--submission_name', default='Submission', type=str, help='the name of submission csv file')
    parser.add_argument('--tta', default=False, action='store_true', help='if use TTA, please add --tta')

    args = parser.parse_args()
    return args


def do_inference(config_dir, work_dir, img_root, epoch, submission_name, tta):
    cfg = Config.fromfile(config_dir)

    # dataset config 수정
    cfg.data.test.classes = classes
    cfg.data.test.img_dir = img_root + 'test'
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 1

    cfg.seed = 2022
    cfg.gpu_ids = [1]
    cfg.work_dir = work_dir

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    # TTA
    if tta:
        cfg.data.test.pipeline[1].img_scale = (512, 512)
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
        cfg.data.test.pipeline[1].flip = True
        cfg.data.test.pipeline[1].flip_direction = ['horizontal', 'vertical']

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=8,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader) # output 계산

    # sample_submisson.csv 열기
    submission = pd.read_csv(SUBMISSION_SAMPLE, index_col=None)

    with open(TEST_JSON, 'r') as json_file:
        images = json.load(json_file)['images']

    output = np.array(output)

    for idx, image in enumerate(output):
        image_name = images[idx]['file_name']
        small_image = image.reshape((1, 256, 2, 256, 2)).max(4).max(2)
        small_image = small_image.flatten()
        
        # submission = submission.append({"image_id" : image_name, "PredictionString" : ' '.join(str(e) for e in small_image.tolist())}, 
        #                             ignore_index=True)
        submission_new_row = pd.DataFrame({"image_id" : [image_name], "PredictionString" : [' '.join(str(e) for e in small_image.tolist())]})
        submission = pd.concat([submission, submission_new_row])
                                    
    submission.to_csv(work_dir + '/' + submission_name + '.csv', index=False)


def main(args):
    do_inference(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)