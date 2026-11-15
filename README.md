# E1 #25: HGB·센서 4/5 전문가와 다중 간격 XGBoost

Uni-D 데이터톤 최종 모델로 팀이 선택한 **#25 E1**의 코드와 저장된 가중치입니다. 최종 파일명은 `submission_e1_s20_mrxgb_train78.csv`, Public Macro F1은 **0.9345095889**입니다. 모델 구조, 피처, 추론 확률 결합 비율을 유지했으며 최종 재현에는 저장된 `weights/`를 사용합니다.

## 예측 실행

```bash
python test.py --input data/battery_data_light/test/test.csv --image_root data/battery_data_light/test --output submission.csv
```

운영진이 요구한 `--input`, `--image_root`, `--output` 세 인자로 실행됩니다. 기본값은 이 폴더의 보존된 `weights/`입니다. GPU나 다른 실험 폴더, 이미지 피처 캐시는 필요하지 않습니다. 입력 CSV의 행 순서대로 `sample_id,label`을 출력하며 라벨은 정수 1~6입니다. 입력의 `label` 열은 예측에 사용하지 않습니다.

대회 데이터와 제출 CSV는 이 저장소에 포함하지 않습니다. `test.py`로 재현해야 하는 원래 #25 제출 CSV의 SHA256은 다음과 같습니다.

```text
273824ac1ff6b3884676b7d0a5733ee3566fa5ff4d4dd16b77cbbeef6a644f57
```

## 학습 실행

E1 #25의 최종 가중치 세 부분은 모두 원래 **Train78 / 33,522행**만 학습했습니다. 같은 구성을 학습하려면 원래 `train.csv`와 Train 이미지를 지정합니다. Validation과 Test를 합치지 않습니다. 학습 스크립트는 지정한 CSV 이외의 데이터 분할을 자동으로 읽지 않습니다.

```bash
# CPU에서 세 모델을 고정 설정으로 학습합니다.
python train.py --input data/battery_data_light/train/train.csv --image_root data/battery_data_light/train --weights_dir retrained_weights --device cpu

# 학습 결과를 별도로 예측합니다. 기본 제출 가중치는 유지됩니다.
python test.py --input data/battery_data_light/test/test.csv --image_root data/battery_data_light/test --output retrained_submission.csv --weights retrained_weights
```

원래 제출 모델의 XGBoost와 전문가 학습에는 CUDA를 사용했습니다. CUDA를 지원하는 별도 환경에서는 같은 명령에 `--device cuda`를 지정합니다. 실행 전에 공유 GPU 사용 여부를 확인해야 합니다. CPU와 CUDA 학습의 수치 연산 및 난수 스트림이 달라, 새 학습 결과가 보존된 가중치와 비트 단위로 같다고 보장하지 않습니다. **제출 파일의 정확한 재현에는 기본 `weights/`를 사용합니다.** 운영진 안내상 `train.py`는 검토용이고 실제 재현 실행은 저장 가중치를 읽는 `test.py`입니다.

`--weights_dir`가 이미 존재하면 학습은 오류로 종료합니다. 보존된 `weights/`와 그 하위 경로에도 학습 결과를 쓰지 못하게 막았습니다. 학습 결과는 다음 구조로 저장하며, 학습 입력의 행·실험 수, 설정, 손실 및 완료 여부를 `training.json`에 남깁니다.

```text
retrained_weights/
  s20/hgb.joblib
  s20/expert.pt
  s20/expert_preprocessing.joblib
  mrxgb/model.joblib
  training.json
```

짧은 실행 점검은 별도 소규모 학습 CSV에 `--smoke`를 붙입니다. 트리는 3회, 전문가는 1 epoch × 2 batch만 학습하므로 결과는 제출용이 아닙니다. 입력의 여섯 클래스가 전문가의 각 관측 뷰에 모두 있어야 합니다.

```bash
python train.py --input small_labeled_train.csv --image_root data/battery_data_light/train --weights_dir smoke_weights --device cpu --smoke
```

## 고정 학습 방법

`training_config.json`에 제출 가중치의 피처 설정과 열 순서를 보존했습니다. 실제 피처 계산은 추론과 같은 로컬 모듈을 사용합니다. 새 학습은 CSV와 실제 PNG를 읽으며 사전 구축 캐시를 요구하지 않습니다.

1. **HGB 앵커:** 기본 788개 피처에 전압 사건 15개를 붙인 803개 피처를 사용합니다. 300회, 깊이 4, 최대 잎 16, 학습률 0.05, L2 5, 시드 42이며 early stopping과 클래스 가중치를 사용하지 않습니다. 결측 중앙값·결측 표시·범주형 one-hot 전처리는 학습 데이터에만 fit합니다. 전압 사건 임계값은 학습 실험의 stride 1·2·3, offset 0 관측에서만 보정합니다.
2. **센서 4/5 전문가:** 센서 파생 92개와 결측 표시를 사용합니다. 20 epoch, epoch당 6,144표본, batch 64, 시드 42입니다. 관측 뷰 선택 확률은 5·10·15초 기준 0.6·0.2·0.2입니다. 클래스별 확률은 0.10·0.15·0.15·0.25·0.25·0.10이며, 클래스 내 실험과 행을 차례로 균등 추출합니다. AdamW 학습률 0.001, weight decay 0.001, cosine 스케줄 최저 0.00001, 센서 잡음 표준편차 0.01, label smoothing 0.02, gradient clipping 2를 사용합니다. 손실은 4/5 CE + 0.2 × 6클래스 보조 CE입니다. 원래 CUDA 학습의 autocast는 bfloat16입니다.
3. **다중 간격 XGBoost:** 기본 788개 피처를 사용합니다. 5초 원본과 10·15초의 모든 offset, 총 여섯 뷰에서 인과 피처를 각각 다시 계산합니다. 원래 행은 총 세 번 등장하므로 뷰 행당 가중치를 1/3로 두어 원래 행당 총 가중치 1과 클래스 비중을 유지합니다. Train78에서는 100,566 학습 행입니다. 전처리는 원래 조밀한 Train 행에만 fit합니다. 300회, 깊이 4, 최대 잎 설정 16, 학습률 0.05, L2 5, subsample·colsample 1, 시드 42, hist 방식을 사용합니다. XGBoost의 기본 grow policy는 원래 코드대로 유지합니다.

최종 추론은 먼저 HGB의 4/5 합계 확률을 보존하면서 전문가가 두 단계에 재분배합니다. 그 결과와 XGBoost의 같은 시점 6클래스 확률을 **0.5:0.5**로 평균합니다. 여러 시점의 확률을 평균하는 방식이 아닙니다. 추가 임계값 탐색이나 새 후처리는 없습니다.

## 인과성 및 이미지 사용

현재 이미지의 밝기·RGB·HSV·2×2 공간 통계 51개를 사용합니다. 기본 센서 및 이미지 밝기 max·p99의 과거 10·30·60·120초 통계를 계산합니다. 시간 계산은 실험 내부 시간 정렬과 실제 관측 시간 차이를 사용하고 입력 순서로 복원합니다. 전체 실험의 미래 최대값이나 평균을 예측 피처로 사용하지 않습니다. 전압 기억과 최고값도 현재까지의 관측만 누적합니다.

식별자는 실험 구분과 결과 복원에만 사용합니다. `sample_id`, `experiment_id`, 이미지 경로, 정답을 예측 피처로 넣지 않습니다. 추론에서는 전처리와 사건 임계값을 다시 fit하지 않습니다. 이미지와 센서가 모두 예측에 기여하며, CNN·ResNet 등의 사전학습 영상 인코더는 이 모델에 사용하지 않습니다. torchvision은 보존된 전문가 모듈의 import 의존성입니다. 외부 사전학습 가중치를 내려받지 않습니다.

## 환경

검증 환경은 Linux, Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, Pillow 12.3.0, joblib 1.6.0, threadpoolctl 3.8.0, XGBoost 3.1.3입니다. 원래 torch 2.11.0+cu128, torchvision 0.26.0+cu128을 사용했으며 자세한 버전은 `environment.json`에 있습니다. 대회 안내에 따라 requirements에서 torch 버전을 고정하지 않았습니다. torch와 torchvision은 서로 호환되는 조합이 필요합니다.

```bash
# 독립된 CPU 추론·학습 환경
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 별도 CUDA 학습 환경에서는 위 파일 대신 requirements-cuda.txt를 사용합니다.
# 이 파일은 xgboost-cpu 대신 같은 버전의 xgboost를 설치합니다.
```

CUDA 학습에는 CUDA 지원 torch 및 GPU가 필요합니다. 두 requirements를 같은 환경에 중복 설치하지 않습니다. 패키지에 포함된 가중치 파일은 모두 25 MB 미만입니다.

## 검증 범위와 성능

- 저장된 최종 가중치로 전체 Test 3,451행을 실제 PNG에서 추론하고 원래 #25 CSV와 바이트 단위로 대조합니다. 반복 실행, 행 셔플, 11개 실험 각각 분리, 실험별 앞 60%만 입력하기, 정답 컬럼 독립성을 자체 점검합니다. 실행 완료 결과는 `verification_summary.json`에 보존합니다.
- 개발 평가는 공통 97실험 5-fold × 두 시드(42, 2026)입니다. E1 Macro F1은 5초 0.896351, 10초 전체 offset 0.894598, 15초 전체 offset 0.894282, mixed500 0.908342입니다. 97실험은 반복해서 확인한 개발 데이터이며 독립 최종 검증이 아닙니다. Public은 0.9345095889이며 Private 점수는 아직 공개되지 않았습니다.
- 원래 패키지의 별도 짧은 학습 점검에서 세 모델의 저장과 새 가중치 추론 경로를 검증했습니다. 원래 Train78 여섯 관측 뷰, 전압 보정값, 전문가 스케일러 및 학습 전처리도 동결된 원본과 대조했습니다. 공개용 포장 과정에서 전체 재학습은 하지 않았습니다.
- 공개용 가중치에서는 추론에 사용하지 않는 `training_groups` 식별자 목록만 제거했습니다. 모델 파라미터·학습된 전처리·숫자 임계값·추론 코드는 유지했습니다. 원본 가중치는 제공 서버에 보존하며 공개용 복사본의 재현 결과를 검사합니다.

자체 검사는 저장소 폴더 밖에서 실행해도 됩니다. `--final`에는 서버에 보존한 원래 #25 CSV를 지정합니다.

```bash
python verify_submission.py --data_dir /path/to/battery_data_light --final /path/to/submission_e1_s20_mrxgb_train78.csv --report /tmp/e1_verification.json
```

`verify_submission.py`는 팀 자체 점검 스크립트이며 주최 측의 `check_submission.py` 원본은 아닙니다. 공식 checker 파일은 현재 제공 패키지에서 확보하지 못했습니다.

## 파일 구성

- `model.py`, `hgb_model.py`, `memory_features.py` 등: 센서·열화상 및 전압 사건 피처와 HGB/전문가 추론.
- `blend_model.py`: 같은 시점의 HGB+4/5 전문가와 다중 간격 XGBoost 확률을 고정 1:1로 결합.
- `train.py`: 제공한 학습 CSV·PNG에서만 전처리와 세 모델을 학습.
- `test.py`: 저장 가중치로만 예측하는 공식 세 인자 CLI.
- `weights/`: 학습된 네 파일. 모든 파일이 25MB 미만.
- `training_config.json`, `environment.json`: 고정 학습 설정과 검증 환경.
- `verification_summary.json`, `submission_info.json`, `artifact_sha256.json`: 자체 점검 집계, 최종 모델 정보, 배포 파일 해시.

대회 데이터·이미지·예측 CSV·내부 실험 로그·인증 정보는 저장소에 포함하지 않습니다. `artifact_sha256.json`은 그 파일 자체와 `.git`을 제외한 배포 파일의 SHA256을 기록합니다.
