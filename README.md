# 기상 초해상도(Super-Resolution) 복원 프로젝트

본 프로젝트는 저해상도 글로벌 기상 예측 모델(**GFS**, 25km 격자) 데이터를 고해상도 미국 국지 예측 모델(**HRRR**, 3km 격자) 데이터를 기반으로 초해상도 복원(Super-Resolution) 및 다운스케일링하는 딥러닝 연구 프로젝트입니다.

---

## 1. 디렉토리 구조 (Directory Structure)

```
기상해상도/
│
├── data/
│   └── weather/
│       ├── raw/                  <- GFS 및 HRRR 다운로드 원본 (.grib2)
│       └── processed/            <- 텍사스 영역 2D 보간 캐싱 텐서 (.npz 및 통계 json)
│
├── src/
│   ├── preprocess.py             <- GFS/HRRR 크롭 및 2D RegularGridInterpolator 보간 모듈
│   ├── build_processed_dataset.py <- GRIB2 ➡️ .npz 텐서 사전 변환 및 캐싱 빌더
│   ├── dataset.py                <- PyTorch Dataset & Normalization 유틸리티
│   └── model.py                  <- WeatherSRResNet (Residual Refinement Super-Resolution Network)
│
├── models/                       <- 훈련된 PyTorch 가중치 (.pt) 및 시각화 이미지 저장
├── download_weather_dataset.py  <- S3 기반 멀티스레드 병렬 다운로더 (경로 자동 인식)
├── train.py                      <- L1 + Gradient Loss 기반 PyTorch 모델 훈련 스크립트
├── evaluate.py                   <- Baseline 대비 RMSE/MAE/PSNR/SSIM 성능 평가 및 시각화
├── run_pipeline.py               <- 엔드투엔드 전체 파이프라인 자동 실행 원클릭 스크립트
└── README.md                     <- 프로젝트 가이드라인
```

---

## 2. 데이터셋 정보 (Dataset Information)

* **대상 기간**: 2023, 2024, 2025년 여름철 (6, 7, 8월)
* **시간 주기**: 6시간 간격 (00z, 06z, 12z, 18z)
* **기상 변수**: 지상 2m 온도 (`TMP:2 m above ground`)
* **해상도 매핑**:
  * **입력값 (Low-Res Input)**: GFS (25km 1D 격자 해상도 ➡️ 3km HRRR 2D 좌표망 보간)
  * **목표값 (High-Res Label)**: HRRR (3km 2D Lambert Conformal 격자)

---

## 3. 기상학적 분석 타깃 영역 (Texas Bounding Box)

미국의 기상이변(스톰, 토네이도, 국지성 폭우)이 가장 다이내믹하게 관찰되는 **텍사스(Texas)주** 영역을 크롭하여 AI 학습 영역으로 사용합니다.

* **위도 (Latitude)**: `25.8` ~ `36.5`
* **경도 (Longitude)**: `-106.6` ~ `-93.5` (0~360 도법 환산 시 `253.4` ~ `266.5`)

---

## 4. 실행 방법

1. **가상환경 활성화 (eccodes / cfgrib 필수)**:
   ```bash
   conda activate weather_env
   ```

2. **전체 파이프라인 원클릭 실행 (데이터 빌드 ➡️ 훈련 ➡️ 평가 및 시각화)**:
   ```bash
   python run_pipeline.py
   ```

3. **개별 모듈 실행**:
   * 데이터셋 캐시 빌드: `python src/build_processed_dataset.py`
   * AI 모델 훈련: `python train.py`
   * 성능 평가 및 그래프 생성: `python evaluate.py`
