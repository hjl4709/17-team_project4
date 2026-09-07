# 1_notes.md — `1.ipynb` 수치 기록

`1.ipynb`와 짝을 이루는 수치 기록 파일. 값을 바꿀 때는 이 파일도 같은 작업 안에서 갱신한다.

## 실행 환경

- torch 2.13.0+cu130 / torchvision 0.28.0+cu130
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 사용)
- `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.backends.cudnn.deterministic=True`, `benchmark=False` (셀 0) — 재현성 개선 시도, 완전한 보장은 아님(아래 참고)

## 데이터 (셀 3, 5)

- 전체 15,515장, 12개 클래스. 최다 `clothes` 5,325장 / 최소 `brown-glass` 607장 → 불균형 비율 8.77배.
- 손상된 파일 0개, 크기 51~888(가로) x 100~936(세로), 색상 모드 RGB 15,481 / P 34

## 중복 이미지 반영 (`2.ipynb` 결과, 셀 7~9)

- 완전 동일 중복 22장 제외(라벨 다른 6장 + 같은 클래스 여분 16장), 근접 중복 25개 그룹은 같은 split에만 배치.
- 분할(70/15/15, cluster 단위, `random_state=42`): train 10,838 / val 2,318 / test 2,337 (총 15,493)

## 전처리 · 불균형 대응 방식 비교 — val 기준 (셀 11~16)

| 설정 | val_macro_f1 |
|---|---|
| 정사각형 리사이즈 + 클래스가중치 (기존) | 0.7227 |
| 비율유지 크롭 + 클래스가중치 | 0.7523 |
| 정사각형 리사이즈 + WeightedRandomSampler | 0.7534 |
| **비율유지 크롭 + WeightedRandomSampler (채택)** | **0.7604** |

## epoch 확장 + 조기 종료 + lr 튜닝 + 스케줄러 (셀 13, 20, 22, 24~26)

- `patience` 기반 조기 종료, epoch 상한(베이스라인 30/freeze 20/unfreeze 15).
- unfreeze lr 튜닝(4epoch, 스케줄러 없이): `{5e-5: 0.9523, 1e-4: 0.9522, 3e-4: 0.9494}` → **5e-5 채택**.
- `ReduceLROnPlateau`: lr=1e-4에서는 효과 없음(3epoch만에 조기 수렴), lr=5e-5에서는 도움됨(12epoch째 lr 감소 후 개선).

## 모델 비교 — 베이스라인 CNN vs 전이학습(freeze) vs 전이학습(unfreeze), val 기준, 최종 설정 (셀 29)

**저장된 best 체크포인트를 val_loader로 재평가한 값:**

| 모델 | val_loss | val_accuracy | val_macro_f1 | val_macro_recall |
|---|---|---|---|---|
| BaselineCNN | 0.8344 | 0.7360 | 0.7095 | 0.7307 |
| EfficientNetV2S(freeze) | 0.5210 | 0.8550 | 0.8009 | 0.8128 |
| **EfficientNetV2S(unfreeze)** | 0.1053 | 0.9728 | **0.9629** | 0.9619 |

## 최종 test 평가 — test는 이 1곳에서만 사용 (셀 31, 32)

- accuracy = 0.9743, macro F1 = 0.9682, macro Recall = 0.9688 — 지금까지 중 최고 기록.
- 클래스별 F1 상위: `green-glass` 1.00, `clothes` 0.99, `trash`/`biological` 0.98
- 클래스별 F1 하위: `plastic` 0.92, `metal` 0.93, `white-glass` 0.95
- 가장 많이 헷갈린 클래스 쌍: (직전 실행 기준) `clothes→shoes`, `white-glass↔plastic` 계열이 지속적으로 상위권.

## 보수적 리뷰에서 발견·처리한 사항 (사용자 요청, 2026-09-05)

### 1. 전처리 비교 셀(`## 2-1`)의 잠재적 버그 — 수정 완료

`## 4` 모델 비교 셀에서 발견했던 "조기 종료 시 마지막 epoch 값 사용" 버그와 동일 패턴이 있었음(`history['val_macro_f1'][-1]`). 3epoch 고정 + 단조 증가라 결과 자체는 우연히 정확했지만, `max(history['val_macro_f1'])`로 고쳐 향후 재발 방지.

### 2. 재현성(run-to-run 편차) — 완화 시도, 완전 검증은 아님

`torch.backends.cudnn.deterministic=True`, `benchmark=False` 추가. 이 설정을 켠 뒤의 실행에서 test macro F1이 **0.9682**로 이전 최고(0.9588)보다도 높게 나왔음 — 다만 **1회 실행 결과일 뿐**이라 "편차가 실제로 줄었다"는 결론은 아직 내릴 수 없음(여러 번 반복 실행해야 확인 가능, 시간상 미실시). 완전한 재현 보장도 아님(정직하게 기록).

### 3. 라벨 노이즈 점검 — 신규 진단 셀 추가, 실제 라벨 오류 발견

### 재현성(run-to-run 편차) — 확인 완료, 실제로 해결됨

`torch.backends.cudnn.deterministic=True`/`benchmark=False` 추가 후 노트북을 **처음부터 두 번 독립적으로 전체 재실행**했다. 두 실행 모두 BaselineCNN/freeze/unfreeze의 val·test 모든 지표가 **소수점 넷째 자리까지 완전히 동일**하게 나왔다(test macro F1 0.9682가 두 번 다 동일, 59건의 라벨 오류 의심 사례 목록도 순서·확신도까지 완전히 일치). cudnn 설정 이전에는 실행마다 0.9529~0.9588 사이에서 흔들렸던 것과 대조적 — **이 파이프라인의 재현성 문제는 실질적으로 해결됐다고 결론.**

### 라벨 노이즈 — 전수 확인 완료(held-out 4,655장 전체 대상)

`## 5-1`에 softmax 확신도 기반 "확신 있게 틀린 예측" 전수 진단 셀 추가(셀 33~36). val+test(held-out) 4,655장 전체에서 확신도 0.9 이상으로 틀린 **59건을 전부** 육안으로 확인했다.

- **`clothes ↔ shoes` 20건(34%) 중 19건이 명백한 라벨 오류** — 운동화·샌들·부츠·로퍼·구두 사진이 `clothes` 폴더에 라벨링되어 있었음. 반대 방향(shoes 폴더에 반바지 사진) 1건.
- 유리 색상 클래스(`white/green/brown-glass`) 혼동 4건 안팎 — 사진상 병 색깔이 라벨과 다르게 보임(조명 가능성 있어 확정적이진 않음).
- 사진이 아닌 이미지 2건 — 마스크 클립아트 일러스트, 볼펜 스케치 그림.
- 나머지 약 33건은 라벨은 맞지만 재질/색상이 실제로 헷갈리는 진짜 어려운 사례이거나 `cardboard`/`paper` 경계가 모호한 경우.

**결론**: 실험 내내 반복된 `clothes→shoes` 오분류의 상당수(20건 중 19건)는 **모델의 한계가 아니라 원본 데이터셋의 라벨 품질 문제**. 라벨 자체를 수정하지는 않음(범위 밖으로 판단, 별도 데이터 큐레이션 작업 필요).

### 해결하지 않고 한계로만 남긴 것

- test 표본 수가 작은 클래스(`brown-glass` 92장 등) — 구조적 한계, 추가 데이터 수집 외 대안 없음.
- 중복 탐지 커버리지 — threshold를 더 풀면 오탐 급증(2.ipynb 토스브러시 사례로 확인됨), 실익보다 위험이 커서 현행 유지.
- 경미한 과적합(train ~99% vs val/test ~97%) — val 기준 조기 종료로 관리되고 있어 추가 조치 불필요로 판단.

## Grad-CAM (셀 38~39)

- 대상: 최종 선정 모델(`best_model`, 재학습 없음), target layer = `best_model.features[-1]`

## 실험 이력 (성능 변화 추적)

| 시점 | 변경 내용 | test accuracy | test macro F1 |
|---|---|---|---|
| 1차 | 기본 파이프라인 (중복 처리 전) | 0.8009 | 0.7402 |
| 2차 | 중복 제거 + leakage 방지 | 0.8019 | 0.7443 |
| 3차 | + 비율유지 크롭 + WeightedRandomSampler | 0.8335 | 0.7728 |
| 4차 | + EfficientNetV2-S unfreeze (고정 4epoch) | 0.9641 | 0.9535 |
| 5차 | + epoch 확장 + 조기 종료 | 0.9671 | 0.9571 |
| 6차 | + 스케줄러(lr=1e-4 고정, 효과 없음) | 0.9649 | 0.9529 |
| 7차 | + lr 튜닝(5e-5) + 스케줄러 | 0.9692 | 0.9588 |
| 8차 | + cudnn 결정론적 설정(2회 재현 확인) | **0.9743** | **0.9682** |
