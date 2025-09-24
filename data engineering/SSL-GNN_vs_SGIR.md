# SSL-GNN을 활용한 폴리머 물성 예측 (SSL-polyGNN) 학습 정리

## 1. 프로젝트 목표 및 배경

본 문서는 Self-Supervised Learning (SSL)을 Graph Neural Networks (GNN)에 적용하여 폴리머(고분자)의 물성을 예측하는 `SSL-polyGNN` 모델을 개발하기 위한 학습 과정을 정리한 것입니다.

초기 학습 과정에서 두 편의 핵심 논문을 분석했습니다.
1.  [cite_start]**SGIR (`Semi-Supervised Graph Imbalanced Regression`)**: 특정 문제, 즉 '불균형한 회귀 데이터'를 해결하는 준지도학습(Semi-supervised) 방법론을 다룬 논문. [cite: 1, 2]
2.  [cite_start]**SSL-GNN 통합 리뷰 논문**: SSL-GNN의 전반적인 사전학습(Pre-training) 방법론을 'Contrastive(대조)'와 'Predictive(예측)' 두 가지 패러다임으로 나누어 설명한 논문. [cite: 813, 820]

## 2. 핵심 개념 분석

### 2.1. SGIR: 특정 문제 해결사 vs. 범용 사전학습

초기에 SGIR이 일반적인 SSL-GNN 사전학습 방법인지 혼동이 있었습니다. 분석 결과, 둘은 목표와 접근 방식이 다릅니다.

- **SGIR (Self-Training에 가까움)**:
    - [cite_start]**목표**: 레이블된 데이터가 특정 값에 몰려있는 '불균형 회귀' 문제를 해결하는 것. [cite: 13, 15]
    - [cite_start]**방식**: 소량의 레이블된 데이터로 시작하여, 레이블 없는 데이터에 **의사 레이블(Pseudo-label)**을 생성하고 데이터셋을 점진적으로 보강하며 학습하는 **자기훈련(Self-training)** 방식입니다. [cite: 15, 16]
    - **결론**: 범용 GNN 인코더를 만드는 사전학습이라기보다는, 특정 다운스트림 태스크(불균형 회귀)를 더 잘 풀기 위한 **준지도학습 솔루션**에 해당합니다.

- **범용 SSL-GNN (Unsupervised Pre-training)**:
    - [cite_start]**목표**: 레이블이 전혀 없는 대규모 데이터로부터 그래프의 구조적, 문맥적 특징을 이해하는 **강력한 범용 GNN 인코더**를 만드는 것. [cite: 833, 834]
    - [cite_start]**방식**: '가짜 문제(Pretext Task)'를 풀어 사전학습을 진행한 뒤, 이 모델을 특정 과제(물성 예측 등)에 맞게 미세조정(Fine-tuning)합니다. [cite: 834]

`SSL-polyGNN`의 목표는 범용 인코더를 만드는 것이므로, 후자인 **Unsupervised Pre-training** 방법론에 집중하는 것이 맞습니다.

### 2.2. SSL-GNN의 두 가지 핵심 사전학습 패러다임

[cite_start]리뷰 논문에 따르면, SSL-GNN의 사전학습은 크게 두 가지 방식으로 나뉩니다. [cite: 820, 860, 861] 두 방법 모두 레이블 없이 모델을 똑똑하게 만드는 것이 목표입니다.

#### **A. Contrastive Learning (대조 학습): "비교하며 배우기"**



- [cite_start]**핵심 아이디어**: **"비슷한 것은 가깝게, 다른 것은 멀게"** 만들어서 모델이 데이터의 본질을 배우게 합니다. [cite: 1037]
- **작동 원리**:
    1.  [cite_start]**뷰(View) 생성**: 하나의 폴리머 그래프에 약간의 변형(원자 특성 마스킹, 결합 제거 등)을 가해 거의 동일한 **'긍정 쌍(Positive Pair)'**을 만듭니다. [cite: 1047, 1048]
    2.  [cite_start]**부정 쌍(Negative Pair) 설정**: 학습 배치 내의 전혀 다른 폴리머 그래프들을 **'부정 쌍'**으로 간주합니다. [cite: 1048]
    3.  [cite_start]**학습**: GNN 인코더가 긍정 쌍의 벡터 표현은 가깝게, 부정 쌍의 벡터 표현은 멀어지도록 학습합니다. [cite: 1037, 1049]
- [cite_start]**장점**: 매우 강력하고 풍부한 표현(representation)을 학습할 수 있습니다. [cite: 1035]
- [cite_start]**단점**: 어떤 '뷰'를 만드느냐(augmentation 전략)에 따라 성능이 크게 좌우되며, 많은 부정 쌍과 비교해야 하므로 계산 비용이 클 수 있습니다. [cite: 1543, 1563]

#### **B. Predictive Learning (예측 학습): "가려진 부분 맞추기"**



- [cite_start]**핵심 아이디어**: **"데이터의 일부를 가리고, 나머지 부분을 이용해 가려진 부분을 예측"**하게 하여 모델이 데이터의 구조를 배우게 합니다. [cite: 863, 1303]
- **작동 원리 (폴리머 예측에 적합한 예시)**:
    1.  [cite_start]**그래프 재구성 (Graph Reconstruction)**: 폴리머 그래프의 일부 원자(node) 종류를 가리거나(Attribute Masking), 화학 결합(edge)을 삭제한 뒤, 모델이 원래대로 복원하도록 학습시킵니다. [cite: 1341, 1379]
    2.  [cite_start]**속성 예측 (Property Prediction)**: 그래프의 내재적 속성(예: 특정 작용기(functional group)의 존재 여부)을 예측하는 과제를 부여합니다. [cite: 1440, 1451]
- [cite_start]**장점**: Pretext task가 직관적이며, 특히 화학 분야의 도메인 지식을 반영하기 용이합니다. [cite: 1557]
- **단점**: 너무 쉬운 예측 문제를 풀 경우, 모델이 깊이 있는 특징을 학습하지 못할 수 있습니다.

## 3. `SSL-polyGNN` 개발 전략

두 가지 패러다임을 모두 이해하는 것은 중요하지만, 초기 개발 단계에서 모두 구현할 필요는 없습니다. 다음의 단계적 접근을 권장합니다.

### **Step 1: Predictive Learning으로 시작하기**

- **추천 방법**: **원자 속성 마스킹 (Node Attribute Masking)**
- **이유**:
    1.  [cite_start]**직관성**: "주변 원자 구조를 보고 가려진 원자 맞추기"는 폴리머의 국소적 화학 환경을 학습하는 데 매우 효과적이고 직관적인 pretext task입니다. [cite: 1246, 1379]
    2.  **구현 용이성**: Contrastive Learning에 비해 구현이 비교적 간단하여 SSL-GNN의 기본 파이프라인을 빠르게 구축하고 테스트해볼 수 있습니다.
    3.  **효과**: 이 방식만으로도 상당한 성능 향상을 기대할 수 있습니다.

### **Step 2: Contrastive Learning으로 성능 극대화하기**

- Predictive 모델로 기본 성능을 확보한 후, 더 높은 성능을 목표로 Contrastive Learning을 도입해볼 수 있습니다.
- [cite_start]**핵심 과제**: 폴리머의 화학적 의미를 해치지 않으면서도 모델에게 유용한 학습 신호를 줄 수 있는 **효과적인 데이터 증강(augmentation) 전략을 설계**하는 것이 중요합니다. [cite: 1300, 1544]

## 4. 요약 및 결론

- **SGIR**은 불균형 데이터 문제 해결을 위한 준지도학습 기법이며, `SSL-polyGNN`이 추구하는 범용 사전학습과는 다릅니다.
- **SSL-GNN 사전학습**은 크게 **Contrastive**와 **Predictive** 방식으로 나뉘며, 둘 다 구현할 필요 없이 하나를 선택하여 시작할 수 있습니다.
- `SSL-polyGNN` 프로젝트는 화학적 직관성이 높고 구현이 용이한 **Predictive Learning (원자 속성 마스킹)**으로 시작하여, 추후 성능 고도화를 위해 **Contrastive Learning**을 도입하는 전략이 효과적입니다.
