# Semantic-segmentation-level2-cv-19
<br />

## 🔍 Overview
### Background
> 바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

<img src='https://camo.githubusercontent.com/86c9fd66258daf9bcaee570f6024589839ca5dfc4efaeaa29f97c9fb82b819a3/68747470733a2f2f692e696d6775722e636f6d2f506e4f6451304c2e706e67' />

> 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.  
> 
> 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.  
> 
> 여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

### Dataset
* 전체 이미지 개수: 3272장
    * Train: 7617장
    * Validation: 655장
* 11 classes: ```Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing```
* Image size: (512x512)
* Coco format의 annotation file
* Pixel 좌표에 따라 카테고리 값을 리턴하여 submission 양식에 맞게 csv 파일을 만들어 제출

### 평가 방법
* Test set의 mIoU (Mean Intersection over Union)로 평가
    * Semantic Segmentation에서 사용되는 대표적인 성능 측정 방법
    * IoU
    * $IoU={|X \cap Y| \over |X \cup Y|}={|X \cap Y| \over {|X|+|Y|-|X \cap Y|}}$
    * mIoU where $c=11$
    * $mIoU={1 \over c} \sum_{c=1}^c{|X_c \cap Y_c|\over|X_c \cup Y_c|}$


<br />

## 📝 Members
- `권혁산` &nbsp; 최종프로젝트 기획안 작성 / 학습 데이터셋 생성 / 모델 학습 
- `김대유` &nbsp; mmSegmentation 환경 설정 / 데이터 셋 정리 / 모델 학습 
- `김찬민` &nbsp; baseline 코드 실행 및 분석
- `이상진` &nbsp; 데이터셋 포맷에 맞게 변경 / mmsegmentation을 통한 모델 학습 / pseudo labelling
- `정효재` &nbsp; baseline 코드 실행 및 분석 / 모델 학습

<br />

## 📃 Code Structure
```
.
├── mmsegmentation
│   └── _custom_configs_
│       ├── deeplabv3_r101_d8
│       ├── hr_city_scape
│       ├── ocr_hr18
│       └── upernet_beit_large_Albu
└── utils
    ├── stratified_kfold
    │   ├── copy_images_kfold.py
    │   ├── create_kfold.py
    │   └── create_mask_kfold.py
    ├── copy_images.ipynb
    ├── create_mask.ipynb
    ├── inference.py
    ├── mask_mmseg_dataset.ipynb
    ├── pseudo_labelling.ipynb
    └── train_val_to_coco.ipynb
```

## Data Structure
```
/opt/ml/input
├── data
│   ├── batch_01_vt
│   ├── batch_02_vt
│   ├── batch_03
│   ├── train.json
│   ├── train_all.json
│   ├── val.json
│   └── test.json
└── mmseg
    ├── annotations
    │   ├── train
    │   └── val
    ├── images
    │   ├── train
    │   └── val
    └── test
```

<br />

## 💻 How to use
### mmsegmentation
```
cd mmsegmentation
python tools/train.py _custom_configs_/{사용할 모델}/model.py
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
● Final Score
  - Public LB score: mIoU 0.8044 (7등)
  - Private LB score: mIoU 0.7551 (5등)
```