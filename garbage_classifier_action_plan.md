# 쓰레기 이미지 분류 모델 (3.ipynb) 개선 실행 계획

> 이 문서는 Claude Code가 `이 md 파일을 읽고 실행해` 라는 지시를 받았을 때, 사람 개입 없이 순서대로 따라갈 수 있도록 작성한 작업 지시서입니다. 각 작업은 독립적으로 실행 가능하지만, **우선순위 순서(1→7)를 지켜서** 진행하세요. 시간이 부족하면 1~2번까지만 해도 됩니다.

## 0. 전제 조건 / 공통 규칙 (모든 작업에 적용)

- 같은 작업 디렉터리에 `3.ipynb`, `split_hj.csv`, `garbage_classification/`(원본 이미지 폴더), 그리고 기존에 저장된 체크포인트(`best_baseline_cnn_v3.pt`, `best_transfer_model_v3.pt`, `unfreeze_lr_5e-05_v3.pt`, `unfreeze_lr_1e-04_v3.pt`, `unfreeze_lr_3e-04_v3.pt`)가 있다고 가정합니다. 체크포인트가 없으면 해당 작업에서 재학습부터 하세요.
- **데이터 분할을 절대 다시 하지 마세요.** `split_hj.csv`의 `train`/`val`/`test` 배정을 그대로 읽어서 재사용합니다 (근접중복 42그룹이 같은 split에 있도록 이미 검증된 상태이므로, 재분할하면 이 보장이 깨집니다).
- **test set은 각 실험당 최종 평가에서 딱 1번만 사용**하세요. 여러 설정을 비교/선택하는 단계에서는 반드시 val만 쓰고, "이걸로 확정" 하기 직전에만 test를 봅니다.
- 새로 만드는 산출물은 전부 `results/` 폴더 아래에 저장하고, 실험을 구분할 수 있게 파일명에 실험명+seed를 포함하세요 (예: `results/seed_repeat/unfreeze_5e-05_seed1_test_metrics.json`). 기존 노트북 결과 파일(`preds_val.csv` 등)을 덮어쓰지 마세요.
- 노트북을 직접 고치기보다는, 가능하면 `3.ipynb`를 복사한 `3_experiments.ipynb` 하나를 만들어 그 안에서 아래 작업들을 셀로 추가해 나가는 걸 권장합니다 (원본 노트북과 결과를 보존하기 위함).
- 각 작업이 끝나면 아래 "완료 기준"을 스스로 체크하고, 결과를 표/숫자 + ppt 발표용 png파일로 남기세요.
- 모든 작업이 끝나면 8번(최종 리포트)을 작성하세요.

---

## 1. [최우선] 시드 반복 실험으로 run-to-run 분산 정량화

**배경**: 지금까지 모든 설정 비교(전처리 그리드, lr 스윕, 모델 비교)가 **각 설정당 1회 학습**으로 이루어졌습니다. `preds_test.csv`로 부트스트랩을 돌려보면 test 표본 구성만으로도 macro F1이 std≈0.004~0.005 정도 흔들리는데, 1.ipynb 대비 정제 효과(+0.0030 acc / −0.0011 macro F1)나 lr 후보 간 차이(5e-5 vs 3e-4: 0.0073)가 이 크기와 비슷하거나 크지 않습니다. 이걸 확인하려면 실제로 여러 번 학습해봐야 합니다.

**지시사항**:

1. 최종 채택 설정(`TransferModel(freeze=False)`, `lr=5e-5`, `use_scheduler=True, scheduler_patience=1`, `patience=3`, `epochs=15`, 데이터 전처리는 기존 최종 설정인 crop+WeightedRandomSampler 그대로)을 **seed = 0, 1, 2, 3, 4** 로 5회 반복 학습하세요.
   - `torch.manual_seed(seed)`, `np.random.seed(seed)`만 바꾸고, **train/val/test 분할(`split_hj.csv`)과 클래스 가중치는 고정**합니다.
   - 매 seed마다 모델 초기화와 `WeightedRandomSampler`/`DataLoader(shuffle=True)`의 무작위성만 달라지게 하면 됩니다.
2. 각 run에서 best checkpoint(val macro F1 기준)로 test set을 1회 평가하고 `accuracy`, `macro_f1`, `macro_recall`을 기록하세요.
3. 결과를 `results/seed_repeat/final_model_seeds.csv` (컬럼: `seed, test_accuracy, test_macro_f1, test_macro_recall`)로 저장하세요.
4. 5개 값의 평균±표준편차를 계산하고, 아래와 비교해서 텍스트로 결론을 남기세요:
   - 1.ipynb 최종 test 수치(accuracy=0.9743, macro_f1=0.9682)와 3.ipynb v3(accuracy=0.9773, macro_f1=0.9671) 차이가 이 표준편차 안에 들어오는지
   - 계산된 표준편차가 이전 부트스트랩 표준편차(acc≈0.003, macro_f1≈0.004~0.005)보다 큰지 (보통 training seed 변동이 test 표본 변동보다 더 큽니다)
5. **시간이 되면**: lr 후보 `1e-4`, `3e-4`도 각각 seed 3회씩 추가로 반복해서, val macro F1 기준 `5e-5`의 우위가 유지되는지 확인하세요. (5e-5 vs 1e-4는 이미 차이가 커서 뒤집힐 가능성이 낮지만, 5e-5 vs 3e-4는 차이가 작아서 뒤집힐 수 있습니다.)

**완료 기준**: `results/seed_repeat/final_model_seeds.csv`가 존재하고, 평균±표준편차와 "정제 효과/lr 선택이 노이즈 안에 있는지"에 대한 한 문단짜리 결론이 있음.

---

## 2. 입력 해상도(IMAGE_SIZE) 실험

**배경**: 현재 `IMAGE_SIZE=128`인데, `EfficientNet_V2_S_Weights.IMAGENET1K_V1`이 사전학습에 사용한 권장 해상도는 이보다 훨씬 큽니다. plastic(recall 0.90), metal(F1 0.93) 등 재질 질감이 중요한 클래스에서 오분류가 지속되는 것과 관련 있을 수 있습니다.

**지시사항**:

1. 먼저 아래 코드로 실제 권장 해상도를 직접 확인하세요 (추측하지 말고 코드로 확인):
   ```python
   from torchvision import models
   w = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
   print(w.transforms())
   ```
2. `IMAGE_SIZE`를 128 → 224로 바꿔서(GPU 메모리가 넉넉하면 300까지) 최종 설정(unfreeze, lr=5e-5, scheduler+조기종료)만 1회(시간 되면 seed 2~3회) 재학습하세요. `CROP_RESIZE = int(IMAGE_SIZE * 1.15)`도 같이 조정됩니다(기존 코드 로직 그대로 사용).
3. 배치 사이즈가 OOM 나면 줄이세요(예: 64→32).
4. val/test macro F1, 그리고 특히 plastic/metal/brown-glass/white-glass 클래스별 F1이 어떻게 바뀌는지 128 기준선과 나란히 표로 비교하세요.
5. 결과를 `results/resolution/IMAGE_SIZE_{128,224,300}_metrics.json`으로 저장하세요.

**완료 기준**: 최소 128 vs 224 두 해상도의 test macro F1 및 클래스별 F1 비교표가 있음.

---

## 3. 전처리·불균형 대응 그리드 재검증

**배경**: 4절의 2×2 그리드(리사이즈 방식 × 불균형 대응)는 **freeze 모델·3epoch짜리 짧은 프록시**로만 순위를 매겼습니다. 1등(crop+sampler, val_macro_f1=0.7640)과 2등(crop+weight, 0.7590) 차이가 0.005로 매우 작아서, lr 스윕에서 벌어졌던 일("짧은 스윕 승자가 최종 학습에선 역전")이 여기서도 일어날 수 있습니다.

**지시사항**:

1. `crop+WeightedRandomSampler`(현재 채택)와 `crop+클래스가중치`(2등) 두 설정만, **최종 학습 설정(unfreeze, lr=5e-5, scheduler+조기종료, 최대 15epoch)** 으로 완전학습해서 직접 비교하세요.
2. val macro F1 기준으로 순위가 3-epoch 프록시 결과와 같은지 다른지 기록하세요.

**완료 기준**: 두 설정의 val/test macro F1 비교 결과와 "순위가 유지됐는지" 결론 한 줄.

---

## 4. Per-sample 예측 저장 + 통계적 유의성 검정 (McNemar / 부트스트랩)

**배경**: 지금은 최종 모델(5e-5)의 per-sample 예측만 저장돼 있어서(`preds_val.csv`, `preds_test.csv`, `probs_val.npy`, `probs_test.npy`), lr 후보 간 paired 비교(McNemar 검정)를 할 수 없습니다.

**지시사항**:

1. 1번 작업에서 학습한 (또는 기존 체크포인트가 남아있다면 그걸 쓴) `lr=1e-4`, `lr=3e-4` 모델에 대해서도 기존 `dump()` 함수와 같은 방식으로 val/test의 `probs_*.npy`와 `preds_*.csv`를 저장하세요 (파일명에 lr 포함: `probs_val_lr1e-04.npy` 등).
2. 아래 McNemar 검정 함수를 구현/사용해서 같은 val set에 대해 `5e-5 vs 1e-4`, `5e-5 vs 3e-4` 쌍을 검정하세요:

   ```python
   import numpy as np
   from scipy.stats import chi2

   def mcnemar_test(true, pred_a, pred_b):
       correct_a = (pred_a == true)
       correct_b = (pred_b == true)
       n01 = np.sum(correct_a & ~correct_b)   # A만 맞음
       n10 = np.sum(~correct_a & correct_b)   # B만 맞음
       if n01 + n10 == 0:
           return 0.0, 1.0
       stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)  # 연속성 보정
       p = 1 - chi2.cdf(stat, df=1)
       return stat, p
   ```

   (참고: McNemar는 "정답을 맞췄는지"만 비교하는 검정이라 macro F1 차이를 직접 검정하진 않습니다. macro F1 차이 자체를 검정하고 싶으면 아래 paired bootstrap을 쓰세요.)

3. Paired bootstrap으로 macro F1 차이의 신뢰구간도 같이 구하세요:

   ```python
   import numpy as np
   from sklearn.metrics import f1_score

   def paired_bootstrap_f1_diff(true, pred_a, pred_b, n_boot=5000, seed=0):
       rng = np.random.RandomState(seed)
       n = len(true)
       diffs = []
       for _ in range(n_boot):
           idx = rng.randint(0, n, n)
           f1_a = f1_score(true[idx], pred_a[idx], average='macro', labels=range(12))
           f1_b = f1_score(true[idx], pred_b[idx], average='macro', labels=range(12))
           diffs.append(f1_a - f1_b)
       diffs = np.array(diffs)
       ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
       return diffs.mean(), ci_low, ci_high
   ```

   신뢰구간이 0을 포함하면 "유의한 차이 없음"으로 해석하세요.

4. 결과(검정통계량, p-value, 신뢰구간)를 `results/significance/lr_comparison.md`에 표로 정리하세요.

**완료 기준**: `5e-5 vs 1e-4`, `5e-5 vs 3e-4` 각각에 대한 McNemar p-value와 paired bootstrap 95% CI가 기록됨.

---

## 5. 확신도(confidence) 보정 분석 확장

**배경**: test set 전체에 대해 10구간 보정 분석을 해보면 confidence 0.9~1.0 구간(전체의 96.6%)은 잘 보정돼 있지만(avg_conf 0.997 vs acc 0.992), **confidence 0.8~0.9 구간은 평균 확신도 0.85인데 실제 정확도는 0.548**로 뚜렷한 과신이 나타납니다. 9절에서는 confidence 0.95 이상 13건만 육안 검토했는데, 그 구간에서도 새 라벨오류 후보가 2건 나왔습니다.

**지시사항**:

1. 아래 calibration 분석 코드를 노트북에 셀로 추가해서 실행하세요 (이미 검증된 로직):

   ```python
   import numpy as np

   def calibration_report(confidence, correct, n_bins=10):
       bins = np.linspace(0, 1, n_bins + 1)
       ece = 0.0
       rows = []
       for i in range(n_bins):
           lo, hi = bins[i], bins[i + 1]
           mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
           n = mask.sum()
           if n == 0:
               continue
           avg_conf = confidence[mask].mean()
           acc_bin = correct[mask].mean()
           ece += (n / len(confidence)) * abs(avg_conf - acc_bin)
           rows.append((lo, hi, n, avg_conf, acc_bin))
       return ece, rows
   ```

   `test_conf`, `(test_preds_all == test_trues_all)`를 입력으로 사용하면 됩니다 (9절에서 이미 계산해둔 변수 재사용 가능).

2. confidence 0.8~0.95 구간에 해당하는 test 오분류 샘플들을 9절과 같은 방식(이미지 그리드)으로 육안 검토하는 셀을 추가하세요. 새로운 라벨오류 후보가 있으면 `label_review_report.md`나 `data_cleaning_summary.md`에 추가할 후보 목록으로 정리하세요.
3. (선택) temperature scaling으로 사후 보정을 시도하고 ECE가 개선되는지 확인하세요.

**완료 기준**: 구간별 calibration 표(ECE 포함)와 0.8~0.95 구간 육안 검토 결과 요약이 노트북/문서에 있음.

---

## 6. 재학습 없이 시도 가능한 저비용 실험

**지시사항** (체크포인트만 있으면 재학습 불필요):

1. **앙상블**: `unfreeze_lr_5e-05_v3.pt`, `unfreeze_lr_1e-04_v3.pt`, `unfreeze_lr_3e-04_v3.pt` 세 모델의 softmax 확률을 평균(soft-voting)해서 val/test macro F1이 단일 최고 모델(5e-5)보다 나아지는지 확인하세요.
2. **Test-time augmentation**: eval 시 원본과 수평 flip 버전의 예측을 평균해서 성능 변화를 확인하세요.
3. **정규화 강화 실험** (재학습 1회): `torch.optim.AdamW(..., weight_decay=1e-4)`와 `nn.CrossEntropyLoss(..., label_smoothing=0.1)`을 적용해서 최종 설정으로 1회 재학습하고, train/val 정확도 갭이 줄어드는지, 그리고 5번에서 발견한 과신(calibration) 문제가 완화되는지 확인하세요.

**완료 기준**: 세 실험 각각의 val/test 지표를 기존 baseline(5e-5 단일 모델)과 비교한 표.

---

## 7. 배경 의존성 정성/정량 점검

**배경**: `clothes913`(모자), `shoes275`(흰 배경 하이힐), `trash640`(칫솔) 등 "흰배경+세로물체" 구도에서 반복적으로 오분류가 나타납니다. 모델이 재질보다 배경 맥락에 의존하고 있을 가능성이 있고, 이는 다른 배경/카메라 환경에서의 일반화 실패로 이어질 수 있습니다.

**지시사항**:

1. test set 오분류 53건과 정분류 샘플들에 대해 간단한 배경 특성(예: 이미지 가장자리 픽셀들의 밝기 표준편차, 또는 배경이 단색에 가까운 정도)을 계산해서, 오분류율과 상관관계가 있는지 간단히 확인하세요. 정교한 배경 분리가 아니어도 되고, 대략적인 heuristic이면 충분합니다.
2. 가능하면 이 데이터셋 밖의 이미지(예: 인터넷에서 구한 실제 촬영 환경의 쓰레기 사진 몇 장) 소량으로 정성적 일반화 테스트를 해보세요.

**완료 기준**: 배경 특성과 오분류 여부의 관계에 대한 간단한 수치/그래프 하나, 또는 외부 이미지 테스트 결과 몇 건.

---

## 8. 최종 리포트 작성

모든 작업(또는 시간 내에 끝낸 만큼)이 끝나면 `results/REPORT.md`를 작성하세요. 포함할 내용:

- 작업별 한 줄 요약 결과 (표 형태 권장: 작업명 | 핵심 수치 | 결론) + 시각화 자료(png)
- "가설검정 결과 통계적으로 유의했던 것" vs "노이즈 안에 있던 것" 구분
- 다음에 우선적으로 더 해볼 만한 것 (남은 시간/자원 기준)
- 이번 실행에서 새로 발견된 라벨오류 후보나 이상치가 있으면 별도로 기록

**완료 기준**: `results/REPORT.md` 존재, 각 섹션에 최소 1문단 이상의 결론 서술.
