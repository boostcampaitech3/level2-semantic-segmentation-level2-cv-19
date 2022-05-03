import os
import json
import shutil

DATASET_PATH = '/opt/ml/input/data'
NEW_PATH = '/opt/ml/input/data/stratified_kfold'

def copy_images(new_dir, JSON_FILE):
    os.makedirs(NEW_PATH + new_dir, exist_ok=True)

    with open(JSON_FILE, 'r', encoding='utf8') as json_file:
        images = json.load(json_file)['images']

    for image in images:
        copy_from = os.path.join(DATASET_PATH, image['file_name'])
        copy_to = NEW_PATH + new_dir + f'/{image["id"]:04}.jpg'
        shutil.copyfile(copy_from, copy_to)

for idx in range(5):
    os.makedirs(NEW_PATH + f'/fold{idx}', exist_ok=True)
    os.makedirs(NEW_PATH + f'/fold{idx}/images', exist_ok=True)
    copy_images(f'/fold{idx}/images/train', os.path.join(NEW_PATH, f'train_fold{idx}.json'))
    copy_images(f'/fold{idx}/images/val', os.path.join(NEW_PATH, f'val_fold{idx}.json'))