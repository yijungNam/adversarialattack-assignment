# Adversarial Attack Assignment

## 실행 방법

1. 모델 학습 및 공격 실행

python test.py

2. epsilon 실험 (보고서용 표 생성)

python epsilon_experiments.py

## 필요한 라이브러리 설치

pip install -r requirements.txt

## 설명

Implements FGSM and PGD adversarial attacks on:

- MNIST
- CIFAR-10 experiments used a torchvision ResNet18 model initialized with ImageNet pretrained weights.
Source: torchvision.models.resnet18 / ResNet18_Weights.DEFAULT
