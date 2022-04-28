import mmcv
from mmcv import Config
from mmseg.datasets import (build_dataloader, build_dataset)
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.transform import resize

######################################
config_dir = '/opt/ml/input/mmsegmentation/_custom_configs_/upernet_r101/model.py'
work_dir = '/opt/ml/input/mmsegmentation/work_dirs/upernet_r101_epoch_40'
img_root = '/opt/ml/input/mmseg/'
epoch = 'latest'
submission_name = 'submission'
######################################

TEST_JSON = '/opt/ml/input/data/test.json'
SUBMISSION_SAMPLE = '/opt/ml/input/code/submission/sample_submission.csv'
classes = ("Background", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile(config_dir)

# dataset config 수정
cfg.data.test.classes = classes
cfg.data.test.img_dir = img_root + 'test'
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 4

cfg.seed = 2022
cfg.gpu_ids = [1]
cfg.work_dir = work_dir

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')
print(checkpoint_path)

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
    
    submission = submission.append({"image_id" : image_name, "PredictionString" : ' '.join(str(e) for e in small_image.tolist())}, 
                                   ignore_index=True)
                                
# submission.to_csv(work_dir + '/submission.csv', index=False)
submission.to_csv(work_dir + '/' + submission_name + '.csv', index=False)