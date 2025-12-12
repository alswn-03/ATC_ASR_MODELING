
from dataload import LibriLoad as LIBRI_DL
from dataload import ATCLoad as ATC_DL
from model import BaseModel
from datetime import datetime

import os
import torch
import torch.nn as nn

print("CUDA AVAIL : ", torch.cuda.is_available())  # True이면 사용 가능
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
NUM_EPOCHS = 50
T_WIDTH = 300
F_MELS = 25
DOWNSAMPLE_FACTOR = 5

PATH = "./model.pth"

AUDIO_FOLDER_PATH = "RiKowalSkipper_01/atc_audio"
CSV_PATH = "RiKowalSkipper_01/labels_final.csv"


CMUdict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
            ]


# train_loader, val_loader = LIBRI_DL(
#     T_WIDTH, 
#     F_MELS, 
#     CMUdict, 
#     DOWNSAMPLE_FACTOR).get_dataloader(BATCH_SIZE)

train_loader, val_loader = ATC_DL(
    AUDIO_FOLDER_PATH, 
    CSV_PATH, 
    T_WIDTH, 
    F_MELS, 
    CMUdict, 
    DOWNSAMPLE_FACTOR).get_dataloader(BATCH_SIZE)

model = BaseModel(len(CMUdict)+1).get_model()
model = model.to(device)

ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if(os.path.exists(PATH)):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for inputs, labels, input_lengths, label_lengths in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)

        # 모델 예측
        outputs = model(inputs)
        outputs = outputs.permute(1, 0, 2)
        # outputs ([71, 8, 40]) <- timestep, batch, class 수

        # CTC Loss 계산
        loss = ctc_loss(outputs, labels, input_lengths, label_lengths)

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 손실 누적
        epoch_loss += loss.item()

    # 에포크당 평균 손실 출력
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train_Loss: {avg_loss:.4f}")


    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels, input_lengths, label_lengths in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            # 모델 예측
            outputs = model(inputs)    
            outputs = outputs.permute(1, 0, 2)
            # outputs ([71, 8, 40]) <- timestep, batch, class 수

            # CTC Loss 계산
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)

            # 손실 누적
            valid_loss += loss.item()

        # 에포크당 평균 손실 출력
        val_avg_loss = valid_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation_Loss: {val_avg_loss:.4f}")

if(os.path.exists(PATH)):
    now = datetime.now()
    NEW_PATH = f"./model_{int(now.timestamp())}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, NEW_PATH)

