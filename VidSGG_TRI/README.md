# 2025.Project.NRF.OTT
(1차년도) 한국연구재단 우수신진연구과제 / OTT 사용자 분석을 위한 데이터 결합 및 증식 기술

고차원적 장면 이해 기반 프레임 태그 추출을 위한 비디오 장면 그래프 생성 고도화 연구

- 비디오 장면 그래프 생성에서의 Temporal Inertia 문제 정의
- 명시적 시간 정보 활용을 위한 Temporal Residual Injection 모듈
- 국소적 시각 정보 활용 능력 극대화를 위한 밀집 표현 기반 특징 추출기 도입



## Requirements

**Recommended Environment**
- OS: Ubuntu 18.04.6 LTS
- CUDA: 11.8
- Python: 3.10
- PyTorch: 2.2.2+cu118
- Torchvision: 0.17.2+cu118
- GPU: 4 × NVIDIA RTX A6000

## Dataset Preparation

본 프로젝트는 Action Genome 데이터셋을 사용합니다. 
다음 [Toolkit](https://github.com/JingweiJ/ActionGenome)을 활용해 데이터셋 다운로드 및 전처리를 수행하고, 다음 링크([files](https://drive.google.com/drive/folders/1tdfAyYm8GGXtO2okAoH1WgVHVOTl1QYe?usp=share_link))에서 제공되는 COCO style의 annotation을 `annotations` 폴더에 위치시켜주세요.
dataset의 디렉토리는 설정은 다음과 같습니다 : 
```
|-- action-genome
    |-- annotations   # gt annotations
        |-- ag_train_coco_style.json
        |-- ag_test_coco_style.json
        |-- ...
    |-- frames        # sampled frames
    |-- videos        # original videos
```

## Train

본 프로젝트는 [[CVPR 2024] OED: Towards One-stage End-to-End Dynamic Scene Graph Generation](https://github.com/guanw-pku/OED?tab=readme-ov-file)을 기반으로 합니다.
학습을 시작하기 위해서는 다음 구글 드라이브([checkpoints](https://drive.google.com/drive/folders/12zh9ocGmbV8aOFPzUfp8ezP0pMTlpzJl?usp=sharing))에서 제공되는 체크포인트를 다룬로드받아 다음과 같이 위치시켜주세요. 
```
exps/params/sgdet/spatial/checkpoint_22_origin.pth
```

학습은 2단계로 나눠서 수행합니다. 먼저 spatial module을 학습하기 위해 다음과 같이 .sh script를 실행시켜주세요. 
```
# Train Spatial Module
sh train_spatial_sgdet_DINOv2.sh
```

이어서, temporal module을 학습하기 위해 다음과 같이 shell script를 실행시켜주세요. 
```
# Train Temporal Module
sh train_temporal_sgdet_DINOv2.sh
```


## Test
학습 완료된 spatial module, temporal module은 각각 다음과 같이 test 할 수 있습니다.
```
# test Spatial Module
python eval_spatial_sgdet_dinov2.py

# test Temporal Module
python eval_temporal_sgdet_dinov2.py
```