# ATC ASR (Air Traffic Control Automatic Speech Recognition)
항공 교통 관제(ATC) 음성을 음소(phoneme) 시퀀스로 변환하는 자동 음성 인식 시스템

CNN과 RNN을 결합한 CRNN 아키텍처와 CTC Loss를 사용하여 음성을 텍스트로 변환합니다.

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [모델 아키텍처](#모델-아키텍처)
- [데이터셋](#데이터셋)
- [설치 방법](#설치-방법)
- [주요 파일 구조](#주요-파일-구조)
- [사용 방법](#사용-방법)
- [모델 상세 설명](#모델-상세-설명)
- [주요 함수 설명](#주요-함수-설명)

---

## 프로젝트 개요

이 프로젝트는 항공 교통 관제 음성을 CMU 음소 사전(39개 음소)으로 변환하는 End-to-End ASR 시스템입니다.

### 주요 특징
- **모델**: CNN + Bi-LSTM (CRNN) 아키텍처
- **손실 함수**: CTC (Connectionist Temporal Classification) Loss
- **입력**: Mel-Spectrogram (300×25 또는 350×25)
- **출력**: 음소 시퀀스 (39개 CMU phonemes + blank token)
- **데이터셋**: RiKowalSkipper_01 ATC 데이터셋, LibriSpeech 데이터셋

---

## 모델 아키텍처

### CRNN 구조
프로젝트는 여러 CRNN 모델 변형을 제공합니다:

```
입력: Mel-Spectrogram (Batch, 1, T_WIDTH, F_MELS)
  ↓
[CNN Block]
  - Conv2D(1→512, kernel=5×8) + BatchNorm + ReLU + MaxPool(2×1)
  - Conv2D(512→512, kernel=5×1) + BatchNorm + ReLU + MaxPool(2×1)
  - Conv2D(512→512, kernel=5×1) + BatchNorm + ReLU
  ↓
[Reshape] → (Batch, Seq_Length, Features)
  ↓
[Dense] → Linear(21504→128) + ReLU
  ↓
[RNN Block]
  - Bi-LSTM(128→256) + BatchNorm
  - Bi-LSTM(512→256) + BatchNorm
  ↓
[Output Layer]
  - Linear(512→256) + ReLU
  - Linear(256→num_classes)
  - LogSoftmax
  ↓
출력: (Batch, Seq_Length, num_classes)
```

### 모델 변형
- **BassCRNN11**: 기본 모델 (ReLU 사용)
- **BassCRNN12**: Clipped ReLU 사용 (max_value=6)
- **BassCRNN21**: BassCRNN11과 동일 (t_width=350)
- **BassCRNN31**: BassCRNN11과 동일 (t_width=400)

---

## 데이터셋

### 1. RiKowalSkipper_01 ATC 데이터셋
- **위치**: `RiKowalSkipper_01/`
- **구성**: 
  - `atc_audio/`: MP3 오디오 파일들
  - `labels_final.csv`: 오디오 파일명, 텍스트, 음소 시퀀스
- **샘플 수**: 약 2,050개
- **형식**: 항공 교통 관제 실제 음성 데이터

### 2. LibriSpeech 데이터셋 (선택적)
- **자동 다운로드**: 코드 실행 시 자동으로 다운로드
- **사용**: 일반 음성 인식 pre-training 용도

### CMU 음소 사전
```python
CMUdict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
           'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
           'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
           'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
```
총 39개 음소 + 1개 blank token (index 0)

---

## 설치 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 주요 패키지
- `torch`: PyTorch 프레임워크
- `torchaudio`: 오디오 처리
- `librosa`: Mel-Spectrogram 생성
- `g2p-en`: 텍스트 → 음소 변환
- `torchmetrics`: WER/CER 계산
- `Levenshtein`: 편집 거리 계산

---

## 주요 파일 구조

```
atc_asr/
├── model.py              # 모델 정의 (BassCRNN11, 12, 21, 31)
├── dataload.py           # 데이터 로더 (ATCLoad, LibriLoad)
├── train.py              # 학습 코드
├── test.py               # 테스트 및 추론 코드
├── requirements.txt      # 패키지 의존성
├── ckpts/                # 학습된 모델 체크포인트
│   ├── 1-1.pth
│   ├── 1-2.pth
│   ├── 2-1.pth
│   ├── 3-1.pth
│   ├── 4-1.pth
│   └── 4-2.pth
└── RiKowalSkipper_01/    # ATC 데이터셋
    ├── atc_audio/        # MP3 오디오 파일
    └── labels_final.csv  # 라벨 데이터
```

---

## 사용 방법

### 1. 학습하기

`train.py` 파일을 열고 사용할 데이터셋을 선택합니다:

```python
# ATC 데이터셋 사용
train_loader, val_loader = ATC_DL(
    AUDIO_FOLDER_PATH, 
    CSV_PATH, 
    T_WIDTH, 
    F_MELS, 
    CMUdict, 
    DOWNSAMPLE_FACTOR
).get_dataloader(BATCH_SIZE)

# 또는 LibriSpeech 데이터셋 사용
# train_loader, val_loader = LIBRI_DL(
#     T_WIDTH, 
#     F_MELS, 
#     CMUdict, 
#     DOWNSAMPLE_FACTOR
# ).get_dataloader(BATCH_SIZE)
```

학습 실행:
```bash
python train.py
```

### 학습 파라미터
- `BATCH_SIZE`: 8
- `NUM_EPOCHS`: 50
- `T_WIDTH`: 300 (시간 축 길이)
- `F_MELS`: 25 (Mel 필터 개수)
- `DOWNSAMPLE_FACTOR`: 5 (풀링으로 인한 다운샘플링 비율)
- `Learning Rate`: 0.001

### 2. 테스트 및 추론하기

```bash
python test.py <model_name>
```

**사용 가능한 모델**:
- `01_1`: t_width=300, 3CNN+2RNN+ReLU
- `01_2`: t_width=300, 3CNN+2RNN+ClippedReLU
- `02_1`: t_width=350, 3CNN+2RNN+ReLU
- `03_1`: t_width=400, 3CNN+2RNN+ReLU
- `04_1`: t_width=300, LibriSpeech로 학습
- `04_2`: t_width=300, LibriSpeech + ATC로 학습

**예시**:
```bash
python test.py 01_1
```

이 명령은:
1. 검증 데이터셋에 대한 WER(Word Error Rate) 계산
2. `for_test.mp3` 파일에 대한 추론 수행
3. 예측된 음소 시퀀스 출력

---

## 모델 상세 설명

### 1. model.py

#### BassCRNN11 클래스
```python
class BassCRNN11(nn.Module):
    def __init__(self, num_classes):
        # num_classes: 40 (39 phonemes + 1 blank)
```

**주요 레이어**:
- `conv1`: Conv2D(1→512, kernel_size=(5,8))
- `conv2`: Conv2D(512→512, kernel_size=(5,1))
- `conv7`: Conv2D(512→512, kernel_size=(5,1))
- `dense`: Linear(21504→128)
- `lstm1`: Bi-LSTM(128→256)
- `lstm2`: Bi-LSTM(512→256)
- `fc1, fc2`: 출력 레이어

**Forward 과정**:
1. Mel-Spectrogram 입력 → CNN으로 특징 추출
2. MaxPooling으로 시간 축 다운샘플링
3. Reshape하여 RNN 입력 형태로 변환
4. Bi-LSTM으로 시퀀스 모델링
5. Fully Connected 레이어로 음소 확률 예측
6. LogSoftmax로 로그 확률 출력

#### BaseModel 클래스
모델 선택을 위한 래퍼 클래스:
```python
model = BaseModel(num_classes=40, model_code="01_1").get_model()
```

### 2. dataload.py

#### ATCLoad 클래스
ATC 데이터셋 로딩 및 전처리:

```python
loader = ATCLoad(
    audio_folder_path="RiKowalSkipper_01/atc_audio",
    csv_path="RiKowalSkipper_01/labels_final.csv",
    t_width=300,
    f_mels=25,
    CMUdict=CMUdict,
    downsample_factor=5
)
```

**주요 기능**:
- `load_data()`: 오디오 파일을 Mel-Spectrogram으로 변환
- `text_transform()`: 음소 문자열을 정수 인덱스로 변환
- `get_dataloader()`: Train/Validation DataLoader 생성 (90:10 분할)

**전처리 과정**:
1. MP3 파일 로드 (16kHz 샘플링)
2. Mel-Spectrogram 생성 (n_fft=1024, hop_length=256)
3. 파워를 dB 스케일로 변환
4. 시간 축을 T_WIDTH로 맞춤 (패딩 또는 리샘플링)
5. 라벨을 정수 인덱스로 변환

#### LibriLoad 클래스
LibriSpeech 데이터셋 로딩 (동일한 전처리 적용):
- 자동으로 데이터를 다운로드하고 캐싱
- G2P(Grapheme-to-Phoneme)로 텍스트를 음소로 변환

#### collate_fn
배치 생성 시 사용되는 함수:
- 가변 길이 시퀀스를 패딩하여 배치로 묶음
- CTC Loss를 위한 input_lengths, label_lengths 계산

### 3. train.py

**학습 루프**:
```python
for epoch in range(NUM_EPOCHS):
    # Training
    for inputs, labels, input_lengths, label_lengths in train_loader:
        outputs = model(inputs)
        loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    with torch.no_grad():
        for inputs, labels, input_lengths, label_lengths in val_loader:
            outputs = model(inputs)
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
```

**체크포인트 저장**:
- 기존 모델이 있으면 타임스탬프를 포함한 새 파일로 저장
- `model_state_dict`와 `optimizer_state_dict` 저장

### 4. test.py

#### ctc_greedy_decoder()
CTC 출력을 음소 시퀀스로 디코딩:
1. 각 타임스텝에서 가장 높은 확률의 음소 선택
2. 연속된 중복 제거 (CTC 규칙)
3. Blank token(0) 제거
4. 인덱스를 음소 문자로 변환

#### test_WER()
검증 데이터셋에 대한 WER 계산:
- 모델 예측 수행
- 예측된 음소와 실제 음소 비교
- Word Error Rate 계산 (torchmetrics 사용)

#### preprocess_infer_audio()
추론을 위한 단일 오디오 파일 전처리:
- 학습 데이터와 동일한 전처리 적용
- 배치 차원 추가

#### infer()
단일 오디오에 대한 추론 수행:
- 모델을 eval 모드로 전환
- Mel-Spectrogram을 입력으로 음소 시퀀스 예측

---

## 주요 함수 설명

### 데이터 처리 함수

#### `text_transform(text_line)`
음소 문자열을 정수 인덱스로 변환
```python
# 입력: "R IH S Z"
# 출력: [27, 16, 28, 37]
```

#### `collate_fn(batch)`
배치 데이터 생성 및 패딩
- 가변 길이 입력/라벨을 동일한 길이로 맞춤
- CTC를 위한 길이 정보 반환

### 디코딩 함수

#### `ctc_greedy_decoder(output, idx_to_phoneme)`
Greedy 방식으로 CTC 출력 디코딩
- 가장 높은 확률의 토큰 선택
- 중복 및 blank 제거

### 평가 함수

#### `test_WER(model, dataloader)`
Word Error Rate 계산
- 전체 검증 데이터셋에 대한 평가
- 예측과 정답 음소 시퀀스 반환

---

## 실행 예시

### 전체 학습 파이프라인
```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 학습 (기본 설정: ATC 데이터셋)
python train.py

# 3. 테스트
python test.py 01_1
```

### 커스텀 오디오 파일 추론
`test.py`의 마지막 부분 수정:
```python
# for_test.mp3 대신 다른 파일 사용
mel_spec = preprocess_infer_audio("your_audio.mp3", t_width)
outputs = infer(model, mel_spec)
result = ctc_greedy_decoder_infer(outputs, idx_to_phoneme)
print("Predicted phonemes:", result)
```

---

## 성능 지표

- **Loss**: CTC Loss (낮을수록 좋음)
- **WER**: Word Error Rate (낮을수록 좋음)
- **평가**: 검증 데이터셋(10%)에 대해 매 에포크마다 측정

---

## 참고 자료

- **CTC Loss**: Connectionist Temporal Classification
- **CMU Pronouncing Dictionary**: 음소 사전
- **Mel-Spectrogram**: 오디오 특징 추출 방법
- **LibriSpeech**: 공개 음성 데이터셋
