import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np

from torchmetrics.text import CharErrorRate
from torchmetrics.text import WordErrorRate
from Levenshtein import distance

from scipy.ndimage import zoom
from model import BaseModel
from dataload import ATCLoad

BATCH_SIZE = 8
NUM_EPOCHS = 50
F_MELS = 25
DOWNSAMPLE_FACTOR = 5
AUDIO_FOLDER_PATH = "RiKowalSkipper_01/atc_audio"
CSV_PATH = "RiKowalSkipper_01/labels_final.csv"

# 명령줄 인자를 확인
if len(sys.argv) != 2:
    print("arg must be one")
    sys.exit(1)

# 인자를 변수로 저장
model_code = sys.argv[1]

if(model_code == "01_1"):
    t_width = 300
    PATH = "ckpts/1-1.pth"
elif(model_code == "01_2"):
    t_width = 300
    PATH = "ckpts/1-2.pth"
elif(model_code == "02_1"):
    t_width = 350
    PATH = "ckpts/2-1.pth"
elif(model_code == "03_1"):
    t_width = 350
    PATH = "ckpts/3-1.pth"
elif(model_code == "04_1"):
    t_width = 300
    PATH = "ckpts/4-1.pth"
else:
    t_width = 300
    PATH = "ckpts/4-2.pth"


print("CUDA AVAIL : ", torch.cuda.is_available())  # True이면 사용 가능
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CMUdict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
            ]

phoneme_to_idx = {p: i+1 for i, p in enumerate(CMUdict)}  # Start indices from 1 (0 is reserved for blank)
idx_to_phoneme = {i+1: p for i, p in enumerate(CMUdict)}
phoneme_to_idx['-'] = 0
idx_to_phoneme[0] = '-'

model = BaseModel(40, model_code).get_model()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 불러오기
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
wer_metric = WordErrorRate()
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model = model.to(device)

def ctc_greedy_decoder(output, idx_to_phoneme): # 가져온 코드
    # 각 시간 스텝마다 가장 높은 확률의 인덱스 선택
    probs = output.exp()
    top_probs, top_labels = probs.topk(1, dim=2)  # top_labels: (Batch=1, Time, 1)

    # 배치와 시간 차원 처리
    top_labels = top_labels.squeeze(2)  # (Batch, Time)
    top_labels = top_labels[0]  # 배치에서 첫 번째 샘플만 사용 (Batch=1인 경우)

    # 연속된 중복 제거 및 블랭크 토큰(0) 제거
    decoded = []
    previous = None
    for label in top_labels:
        label = label.item()  # 텐서를 스칼라 값으로 변환
        if label != previous and label != 0:
            decoded.append(label)
        previous = label

    # 인덱스를 음소로 매핑
    phonemes = [idx_to_phoneme.get(idx, '<UNK>') for idx in decoded]

    return phonemes

def ctc_greedy_decoder_infer(output, phoneme): # 가져온 코드
    # output: (Batch=1, Time, num_classes)
    # num_classes에는 CTC의 블랭크 토큰이 포함되어 있어야 합니다.

    # 각 시간 스텝마다 가장 높은 확률의 인덱스 선택
    probs = output.exp()
    top_probs, top_labels = probs.topk(1, dim=2)  # top_labels: (Batch=1, Time, 1)

    top_labels = top_labels.squeeze(0).squeeze(1)  # (Time,)

    # 연속된 중복 제거 및 블랭크 토큰(0) 제거
    decoded = []
    previous = None
    for label in top_labels:
        if label != previous and label != 0:
            decoded.append(label.item())
        previous = label

    # 인덱스를 음소로 매핑
    phonemes = [idx_to_phoneme.get(idx, '<UNK>') for idx in decoded]

    return phonemes

# test 및 CER 출력
def test_WER(model, dataloader): 
    model.eval()
    preds, targets = [], []
    wer_arr=[]
    with torch.no_grad():
        for inputs, labels, input_lengths, label_lengths in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            # infer(model, mel_spec)
            outputs = model(inputs)
            outputs = outputs.permute(1, 0, 2)
            # timestep, batch, class 수

            # 디코딩하여 음소 시퀀스 획득
            phonemes = ctc_greedy_decoder(outputs, idx_to_phoneme) # ['R', 'IH', 'S', 'Z', 'S', 'IH', 'K', 'S', 'IH', 'R', 'IH', 'Z', 'B', 'ZH', 'OY', 'ZH', 'OY']
            # integer였던 labels를 phonemes 형태로
            labels_phonemes = [idx_to_phoneme[label.item()] for label in labels]

            single_phonemes = [" ".join(phonemes)] # ['R IH S Z S IH K S IH R IH Z B ZH OY ZH OY']
            single_labels = [" ".join(labels_phonemes)]
            #print(single_phonemes)
            #print(single_labels)

            preds.extend(single_phonemes)
            targets.extend(single_labels)

    # WER 업데이트
    wer_metric.update(preds, targets)

    # WER 계산
    wer =  wer_metric.compute()
    print(f"Word Error Rate: {wer:.2f}")

    return wer, preds, targets

def preprocess_infer_audio(audio_path, t_width):
    audio_file = os.path.join(audio_path)
    sig, sr = librosa.load(audio_file, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=1024, hop_length=256, n_mels=25) # (height, width)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = np.expand_dims(mel_spec.T, axis=-1)
    # print(os.path.basename(audio_file), "mel_spec_db.shape : ", mel_spec.shape)

    current_width = mel_spec.shape[0]

    if current_width < t_width:
        pad_width = t_width - current_width
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        mel_spec = np.pad(
            mel_spec,
            pad_width=((left_pad, right_pad), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )

    elif current_width > t_width:
        scale_factor = t_width / current_width
        mel_spec = zoom(mel_spec, (scale_factor, 1, 1), order=1)

    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
    mel_spec = mel_spec.unsqueeze(0)
    mel_spec = mel_spec.permute(0, 3, 1, 2)
    return mel_spec

def infer(model, mel_spec):
    with torch.no_grad():
        mel_spec = mel_spec.to(device)
        # 모델 순전파
        outputs = model(mel_spec)  # outputs: (Time, Batch=1, num_classes)

        # Time과 Batch 차원 순서 변경
        outputs = outputs.permute(1, 0, 2)  # (Batch=1, Time, num_classes)

        return outputs


train_loader, val_loader = ATCLoad(
    AUDIO_FOLDER_PATH, 
    CSV_PATH, 
    t_width, 
    F_MELS, 
    CMUdict, 
    DOWNSAMPLE_FACTOR).get_dataloader(BATCH_SIZE)

wer, preds, targets=test_WER(model, val_loader)
# werT, predsT, targetsT=test_WER(model, test_dataloader)
mel_spec = preprocess_infer_audio("for_test.mp3", t_width)
outputs = infer(model, mel_spec)
result = ctc_greedy_decoder_infer(outputs, idx_to_phoneme)
print(result)
