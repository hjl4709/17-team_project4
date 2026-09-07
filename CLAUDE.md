# CLAUDE.md

## 프로젝트 개요

- **목표**: `garbage_classification/` 폴더에 클래스별 하위 폴더로 정리된 쓰레기 이미지(jpg, 총 15,515장, 12개 클래스)를 학습해 이미지를 자동 분류하는 딥러닝(CNN) 모델을 설계·구현·평가한다.
- **데이터 구조**: `garbage_classification/<클래스명>/*.jpg` — 폴더명이 곧 라벨. 클래스 목록: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass` (총 12종).
- **타겟 변수**: 이미지가 속한 폐기물 클래스 (12종 다중 분류, multiclass)
- **데이터 특성**: 클래스별 이미지 수 불균형 있음 (`clothes` 5,325장 vs `brown-glass` 607장, 최대/최소 비율 약 8.8배) → Accuracy 단독 판단 금지, macro F1 / macro Recall / 클래스별 Confusion Matrix를 우선 고려. 이미지 크기·비율은 파일마다 제각각(RGB) → 리사이즈 필요.
- **산출물**: 재현 가능한 코드 (Jupyter Notebook), 아래 "코드 스타일 및 사용 가능 기법 제한" 규칙을 따름
- **개인 프로젝트**: 팀 구성/역할 분담/발표는 해당 없음

## 응답 언어

- Claude Code는 이 프로젝트의 모든 응답(설명, 커밋 메시지, 로그, 질문 등)을 **한국어**로 작성한다.
- 코드 내 변수명·함수명 등은 관례상 영어를 유지해도 된다.

## 코드 스타일 및 사용 가능 기법 제한 (중요)

- 이 프로젝트의 모든 코드는 `01.ipynb` ~ `07.ipynb`(PyTorch/CNN 실습 노트북)에서 실제로 등장한 코드 스타일과 기법을 기준으로 작성한다. 아직 배우지 않은 것으로 간주되는 기법(예: 이전 HR Attrition 프로젝트에서만 쓰인 `Pipeline`/`ColumnTransformer`/`GridSearchCV`/`RandomizedSearchCV`/`SHAP`, 또는 아예 다뤄본 적 없는 `albumentations`, `Optuna`, `wandb`, `torch.cuda.amp`, DDP/멀티GPU, ONNX 변환 등)는 임의로 도입하지 않는다.
- `01~07.ipynb`에 없는 라이브러리·함수·기법을 **꼭 써야 하는 경우**, 해당 코드 셀 바로 위(또는 안)에 `# [NEW] 01~07.ipynb에 없는 기법: <이름> — <왜 필요한지>` 형식의 주석을 남기고, `PROGRESS.md`의 "결정 근거"에도 동일 내용을 기록한다. 이 표시 없이 새 기법을 조용히 섞어 쓰지 않는다.
- **01~07.ipynb 기준으로 이미 학습된(사용 가능한) 것들**:
  - 기본: `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `numpy`, `matplotlib.pyplot`, `seaborn`, `time`, `os`, `copy`, `glob`, `tqdm.auto`, `platform`/`matplotlib.font_manager`(한글 폰트 설정 — Windows는 `Malgun Gothic`)
  - 데이터: `torchvision.datasets`(`MNIST`, `CIFAR10`, `ImageFolder`), `torchvision.transforms`(`Resize`, `RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `RandomAffine`, `ColorJitter`, `ToTensor`, `Normalize`), `torch.utils.data`(`Dataset`, `DataLoader`, `TensorDataset`, `Subset`, `random_split`), `PIL.Image`, `cv2`(`imread`/`cvtColor`/`resize`)
  - 모델 구성 요소: `nn.Linear`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.BatchNorm1d`/`BatchNorm2d`, `nn.MaxPool2d`, `nn.Dropout`/`Dropout2d`, `nn.ReLU`, `nn.Flatten`, `nn.AdaptiveAvgPool2d`, `nn.Sequential`, `nn.Module` 직접 서브클래싱, Residual/BasicBlock(ResNet 직접 구현), U-Net 직접 구현(EncoderBlock/DecoderBlock 패턴)
  - 전이학습: `torchvision.models`(예: `efficientnet_v2_s` + `IMAGENET1K_V1` 가중치, 백본 `freeze`), `timm.create_model`(예: `resnetv2_50`, `pretrained=True`, `num_classes=0`, `global_pool='avg'`)
  - 학습 루프: 직접 구현한 for-epoch/for-batch 루프(`optimizer.zero_grad()` → forward → loss → `backward()` → `step()`), 직접 구현한 `evaluate()`/`train_model()` 헬퍼 함수, 검증 지표 기준 best 체크포인트를 `torch.save(model.state_dict(), ...)`로 저장
  - 옵티마이저/스케줄러: `optim.SGD`, `optim.Adam`, `StepLR`/`ReduceLROnPlateau`/`CosineAnnealingLR`
  - 손실함수: 다중분류는 `nn.CrossEntropyLoss`, 이진분류 예시는 `nn.BCEWithLogitsLoss`(`pos_weight`로 불균형 대응)
  - 가중치 초기화: `nn.init.kaiming_normal_`/`xavier_normal_`/`normal_`
  - 클래스 불균형 대응: `Counter`로 분포 확인, `pos_weight`/클래스 가중치 계산(`1/count` 기반)
  - 평가지표: `sklearn.metrics`의 `confusion_matrix`, `classification_report`, `roc_curve`, `auc`, `accuracy_score`, `recall_score`, `precision_score`, `roc_auc_score`, `balanced_accuracy_score`, `average_precision_score`; `seaborn.heatmap` 또는 `plt.imshow`+`itertools`로 confusion matrix 시각화
  - threshold 튜닝(이진분류 예시): `np.arange`로 threshold 스윕 후 지표 비교
  - 설명가능성: Grad-CAM 직접 구현(forward/backward hook)
  - 참고만 하고 실제 사용 여부는 셀별로 확인: `torchsummary.summary`, `torch.nn.utils.prune`(04.ipynb에 import만 되어 있고 실사용 예시는 없음 — 쓰려면 위 표시 규칙 적용)
- 12개 클래스 다중분류이므로 `nn.CrossEntropyLoss` + `argmax` 기반 예측(MNIST/CIFAR10 실습과 동일한 패턴)이 기본 노선이며, 이진분류(`BCEWithLogitsLoss`+`sigmoid`)용으로 만들어진 헬퍼 함수(`evaluate`, `train_model` 등)를 그대로 재사용할 수 없다 — 다중분류에 맞게 고쳐 써야 한다.

## 진행상황 기록 규칙 (중요)

- 작업 진행 상황, 결정 사항, 코드 변경 내역은 이 CLAUDE.md에 직접 기록하지 않는다.
- 별도 파일 **`PROGRESS.md`** 에 아래 형식으로 append 방식으로 기록한다.
- 코드/설계에 영향을 주는 작업을 마칠 때마다 PROGRESS.md 갱신을 빠뜨리지 않는다.

```
## YYYY-MM-DD
- 작업: 이번에 진행한 작업 요약
- 변경사항: 추가/수정/삭제된 코드나 파일
- 결정 근거: 왜 이렇게 했는지 (여러 방법 중 선택한 이유, 01~07.ipynb에 없는 기법을 썼다면 그 이유 포함)
- 다음 단계: 이어서 할 작업
```

## 코딩 원칙 (karpathy-guidelines)

1. **코딩 전에 생각하기**: 가정을 명시적으로 말한다. 불확실하면 임의로 진행하지 말고 질문한다. 여러 해석이 가능하면 선택지를 제시한다. 더 단순한 방법이 있으면 먼저 말한다.
2. **단순함 우선**: 요청받은 것 이상을 구현하지 않는다. 당장 쓰이지 않는 유연성·추상화·예외처리를 추가하지 않는다.
3. **외과적 수정**: 요청과 무관한 코드/스타일/포맷은 건드리지 않는다. 기존 스타일을 따른다. 정리가 필요해 보이는 다른 코드를 발견하면 삭제하지 말고 언급만 한다.
4. **목표 기반 실행**: 각 작업을 검증 가능한 성공 기준으로 바꾼다. (예: "리사이즈/증강 적용 전후 클래스별 샘플 이미지로 확인" "train/val loss curve로 과적합 여부 확인") 여러 단계 작업은 실행 전 `단계 → 검증 방법` 형태로 짧게 계획을 제시한다.

## 실험 비교 규칙 (test set 보호)

- 모델 구조·하이퍼파라미터·데이터 증강·불균형 대응·threshold를 **비교하거나 선택하는**
  모든 셀은 train/validation 분할(검증 데이터, `val_loader`/`val_set`)로만 평가한다.
  `test_loader`/`test_set`이 비교 목적 셀에 등장하면 그 셀은 규칙 위반이다.
- `test_loader`/`test_set`은 노트북 전체에서 **최종 모델 평가 셀 1곳**에서만 쓴다.
- "test 점수가 더 높으니 A를 쓰자"는 판단 근거로 쓰지 않는다.
  비교는 검증 점수(`val_loss`/`val_accuracy`/`val_f1` 등)로만 한다.
- 데이터 규모와 학습 비용을 고려해 StratifiedKFold 교차검증 대신 고정된 train/val/test 분할(예: 70/15/15, `stratify`로 클래스 비율 유지 + `random_state` 고정)을 써도 된다.
- 이미 test로 고른 설정을 발견하면, 되돌리지 말고 validation으로 다시 골라
  결론이 바뀌는지 확인한 뒤 기록한다.

## 모델 일관성 규칙

- 전처리(리사이즈·정규화·증강 포함)는 노트북에서 **한 번만** 정의하고,
  학습·튜닝·평가·해석이 모두 같은 transform/데이터를 쓴다. train 전용 증강과 eval용 transform은 분리하되 통계값(정규화 mean/std)은 동일하게 유지한다.
- 예측 근거 설명(Grad-CAM 등)은 **최종 성능을 낸 그 모델**에서 뽑는다.
  해석용으로 별도 재학습하지 않는다.

## 문서 동기화 규칙

- 메인 분석 노트북을 수정하면 같은 작업 안에서 그 노트북과 짝을 이루는 `*_notes.md` 수치도 갱신한다.
- 수치를 기록할 때는 그 값을 낸 셀 번호나 변수명을 함께 적는다.
- 라이브러리 버전이 다르면 결과가 달라질 수 있으므로,
  수치를 기록할 때 실행 환경(torch/torchvision 버전, GPU 사용 여부)을 함께 남긴다.

## 프로젝트 진행 단계

### 1. 문제 정의

- 타겟(12개 폐기물 클래스)과 데이터 출처(`garbage_classification/<클래스>/*.jpg`, 총 15,515장) 명시
- 성공 기준(어떤 지표를 얼마나 개선하면 성공인지) 문서화
- 클래스 불균형 여부 사전 확인(클래스별 이미지 수 집계)

### 2. 탐색적 데이터 분석 (EDA)

- 클래스별 이미지 수 분포 확인 (막대그래프 → 불균형 여부가 지표/샘플링 전략에 직결)
- 클래스별 샘플 이미지 시각화
- 이미지 크기·가로세로 비율·색상 모드(RGB) 확인, 손상되거나 열리지 않는 파일 여부 점검
- (필요 시) 클래스 간 시각적으로 헷갈리기 쉬운 쌍(예: 유리 색상별 클래스) 확인

### 3. 데이터 전처리 & 피처 엔지니어링

- 데이터 로드 방식 선택 및 근거 기록 (`ImageFolder` vs 직접 구현 `Dataset`)
- train/val/test 분할 방식과 비율, `stratify` 적용 여부 및 근거 기록
- 리사이즈 크기, 정규화 통계(ImageNet 사전학습 통계 사용 여부), train 전용 증강 종류 및 근거 기록
- 클래스 불균형 대응 전략 선택 및 근거 기록 (클래스 가중치, `WeightedRandomSampler`, threshold 조정 등)
- **Data Leakage 방지**: train/val/test 분리 이후에 증강·정규화 통계 적용(통계 자체는 고정값 또는 train 기준으로만 산출)

### 4. 모델링

- 베이스라인 CNN(처음부터 학습, 얕은 구조)부터 시작 → 이후 비교 기준점 확보
- 전이학습 모델(사전학습 백본, freeze/unfreeze 비교) 등 최소 2개 이상 모델 비교
- 검증셋 기준으로 성능 안정적으로 측정 (데이터가 크면 고정 분할, 가능하면 StratifiedKFold도 고려)
- 하이퍼파라미터(학습률/에폭/옵티마이저 등) 튜닝은 검증 점수 기준으로 진행
- 실험은 조건을 바꿔가며 모델/파라미터/지표를 표로 로그 기록

### 5. 모델 평가

- 단순 Accuracy만으로 판단하지 않음 (클래스 불균형이 있으므로 왜곡 가능)
- macro F1-score, macro Recall 등 다중분류·불균형에 맞는 지표 함께 사용
- 12x12 Confusion Matrix로 클래스별 오분류 패턴 확인 (어떤 클래스끼리 헷갈리는지)
- 여러 모델의 지표를 표로 비교하고 최종 모델 선정 근거 명시

### 6. 결과 해석

- Grad-CAM 등으로 모델이 이미지의 어느 영역을 근거로 예측했는지 시각화
- 어떤 클래스가 특히 잘 맞고 어떤 클래스끼리 자주 혼동되는지 도메인 관점에서 해석 (예: 분리수거 실무에 주는 시사점)
- 모델의 한계와 개선 방향 기록

## 공통 유의사항

- **재현성**: `random_state`/`torch.manual_seed`(seed) 고정, 사용 라이브러리 버전(torch/torchvision 등) 명시
- **데이터 분할**: train/validation/test 분리 원칙 준수 (test는 최종 평가 전까지 사용하지 않음)
- **과적합 방지**: train/val loss·accuracy curve를 반드시 확인하고 기록
- **버전 관리**: Git으로 코드 이력 관리(아직 git 저장소가 아니면 필요 시 `git init` 안내)

## 최종 점검 체크리스트

- [ ] 문제 정의 및 데이터 출처(클래스 수/이미지 수)가 명시되어 있다
- [ ] EDA 핵심 결과(클래스별 이미지 수 분포, 샘플 이미지)가 시각화와 함께 제시되어 있다
- [ ] 전처리 과정(리사이즈/정규화/증강)과 근거가 설명되어 있다
- [ ] 최소 2개 이상의 모델(베이스라인 CNN vs 전이학습 등)을 비교했다
- [ ] 검증 결과가 포함되어 있다
- [ ] 클래스 불균형을 고려한 평가지표(Accuracy 외 macro F1/Recall, Confusion Matrix)를 사용했다
- [ ] 결과에 대한 도메인 관점의 해석이 포함되어 있다
- [ ] 코드가 재현 가능한 형태(노트북)로 제공된다
- [ ] 01~07.ipynb에 없는 기법을 썼다면 `# [NEW]` 표시와 PROGRESS.md 기록이 되어 있다
