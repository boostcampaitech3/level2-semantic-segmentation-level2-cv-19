# Semantic-segmentation-level2-cv-19
<br />

## ๐ Overview
### Background
> ๋ฐ์ผํ๋ก ๋๋ ์์ฐ, ๋๋ ์๋น์ ์๋. ์ฐ๋ฆฌ๋ ๋ง์ ๋ฌผ๊ฑด์ด ๋๋์ผ๋ก ์์ฐ๋๊ณ , ์๋น๋๋ ์๋๋ฅผ ์ด๊ณ  ์์ต๋๋ค. ํ์ง๋ง ์ด๋ฌํ ๋ฌธํ๋ '์ฐ๋ ๊ธฐ ๋๋', '๋งค๋ฆฝ์ง ๋ถ์กฑ'๊ณผ ๊ฐ์ ์ฌ๋ฌ ์ฌํ ๋ฌธ์ ๋ฅผ ๋ณ๊ณ  ์์ต๋๋ค.

<img src='https://camo.githubusercontent.com/86c9fd66258daf9bcaee570f6024589839ca5dfc4efaeaa29f97c9fb82b819a3/68747470733a2f2f692e696d6775722e636f6d2f506e4f6451304c2e706e67' />

> ๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ ์ค ํ๋์๋๋ค. ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์๋๋ค.  
> 
> ๋ฐ๋ผ์ ์ฐ๋ฆฌ๋ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ Detection ํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํด๋ณด๊ณ ์ ํฉ๋๋ค. ๋ฌธ์  ํด๊ฒฐ์ ์ํ ๋ฐ์ดํฐ์์ผ๋ก๋ ์ผ๋ฐ ์ฐ๋ ๊ธฐ, ํ๋ผ์คํฑ, ์ข์ด, ์ ๋ฆฌ ๋ฑ 10 ์ข๋ฅ์ ์ฐ๋ ๊ธฐ๊ฐ ์ฐํ ์ฌ์ง ๋ฐ์ดํฐ์์ด ์ ๊ณต๋ฉ๋๋ค.  
> 
> ์ฌ๋ฌ๋ถ์ ์ํด ๋ง๋ค์ด์ง ์ฐ์ํ ์ฑ๋ฅ์ ๋ชจ๋ธ์ ์ฐ๋ ๊ธฐ์ฅ์ ์ค์น๋์ด ์ ํํ ๋ถ๋ฆฌ์๊ฑฐ๋ฅผ ๋๊ฑฐ๋, ์ด๋ฆฐ์์ด๋ค์ ๋ถ๋ฆฌ์๊ฑฐ ๊ต์ก ๋ฑ์ ์ฌ์ฉ๋  ์ ์์ ๊ฒ์๋๋ค. ๋ถ๋ ์ง๊ตฌ๋ฅผ ์๊ธฐ๋ก๋ถํฐ ๊ตฌํด์ฃผ์ธ์! ๐

### Dataset
* ์ ์ฒด ์ด๋ฏธ์ง ๊ฐ์: 3272์ฅ
    * Train: 7617์ฅ
    * Validation: 655์ฅ
* 11 classes: ```Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing```
* Image size: (512x512)
* Coco format์ annotation file
* Pixel ์ขํ์ ๋ฐ๋ผ ์นดํ๊ณ ๋ฆฌ ๊ฐ์ ๋ฆฌํดํ์ฌ submission ์์์ ๋ง๊ฒ csv ํ์ผ์ ๋ง๋ค์ด ์ ์ถ

### ํ๊ฐ ๋ฐฉ๋ฒ
* Test set์ mIoU (Mean Intersection over Union)๋ก ํ๊ฐ
    * Semantic Segmentation์์ ์ฌ์ฉ๋๋ ๋ํ์ ์ธ ์ฑ๋ฅ ์ธก์  ๋ฐฉ๋ฒ
    * IoU
    * $IoU={|X \cap Y| \over |X \cup Y|}={|X \cap Y| \over {|X|+|Y|-|X \cap Y|}}$
    * mIoU where $c=11$
    * $mIoU={1 \over c} \sum_{c=1}^c{|X_c \cap Y_c|\over|X_c \cup Y_c|}$


<br />

## ๐ Members
- `๊ถํ์ฐ` &nbsp; ์ต์ขํ๋ก์ ํธ ๊ธฐํ์ ์์ฑ / ํ์ต ๋ฐ์ดํฐ์ ์์ฑ / ๋ชจ๋ธ ํ์ต 
- `๊น๋์ ` &nbsp; mmSegmentation ํ๊ฒฝ ์ค์  / ๋ฐ์ดํฐ ์ ์ ๋ฆฌ / ๋ชจ๋ธ ํ์ต 
- `๊น์ฐฌ๋ฏผ` &nbsp; baseline ์ฝ๋ ์คํ ๋ฐ ๋ถ์
- `์ด์์ง` &nbsp; ๋ฐ์ดํฐ์ ํฌ๋งท์ ๋ง๊ฒ ๋ณ๊ฒฝ / mmsegmentation์ ํตํ ๋ชจ๋ธ ํ์ต / pseudo labelling
- `์ ํจ์ฌ` &nbsp; baseline ์ฝ๋ ์คํ ๋ฐ ๋ถ์ / ๋ชจ๋ธ ํ์ต

<br />

## ๐ Code Structure
```
.
โโโ mmsegmentation
โ   โโโ _custom_configs_
โ       โโโ deeplabv3_r101_d8
โ       โโโ hr_city_scape
โ       โโโ ocr_hr18
โ       โโโ upernet_beit_large_Albu
โโโ utils
    โโโ stratified_kfold
    โ   โโโ copy_images_kfold.py
    โ   โโโ create_kfold.py
    โ   โโโ create_mask_kfold.py
    โโโ copy_images.ipynb
    โโโ create_mask.ipynb
    โโโ inference.py
    โโโ mask_mmseg_dataset.ipynb
    โโโ pseudo_labelling.ipynb
    โโโ train_val_to_coco.ipynb
```

## Data Structure
```
/opt/ml/input
โโโ data
โ   โโโ batch_01_vt
โ   โโโ batch_02_vt
โ   โโโ batch_03
โ   โโโ train.json
โ   โโโ train_all.json
โ   โโโ val.json
โ   โโโ test.json
โโโ mmseg
    โโโ annotations
    โ   โโโ train
    โ   โโโ val
    โโโ images
    โ   โโโ train
    โ   โโโ val
    โโโ test
```

<br />

## ๐ป How to use
### mmsegmentation
```
cd mmsegmentation
python tools/train.py _custom_configs_/{์ฌ์ฉํ  ๋ชจ๋ธ}/model.py
```

<br />

## Evaluation
### Models

|Decoder|Backbone|Parameter|LB mIoU|
|:--:|:--:|:--:|:--:|
|FCN|ResNet-50|baseline|0.5141|
|UNet++|Conv block|baseline|0.2181|
|HRNet-18|OCRNet-18|epoch 40, albumentations, TTA|0.6148|
|Deeplabv3|ResNet-101|epoch 50, albumentations, TTA|0.6429|
|UperNet|BEiT-L|epoch 50, albumentations, multi-scale, TTA|0.8008|
|UperNet|Swin-L|epoch 50, albumentations, TTA|0.7054|
|UperNet|BEiT-L|epoch 50, albumentations, TTA|0.7773|
|UperNet|BEiT-L|epoch 50, albumentations, TTA, CLAHE|0.7671|
|Senformer|Swin-L|epoch 50, albumentations, TTA|0.6829|

### Ensemble

```Upernet BEiT-L``` + ```Upernet Swin-L``` = 0.8044

```
โ Final Score
  - Public LB score: mIoU 0.8044 (7๋ฑ)
  - Private LB score: mIoU 0.7551 (5๋ฑ)
```