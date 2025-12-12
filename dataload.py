import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import os
import re
import pandas as pd
import librosa
import pickle
import numpy as np
from g2p_en import G2p
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn (batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.int32)
    # inputs : (8, 300, 25, 1)
    # labels : 8개의 labels
    # label_lengths : torch.Size([8])

    # 입력 시퀀스를 패딩
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    inputs = inputs.permute(0, 3, 1, 2)
    # inputs : torch.Size([8, 1, 300, 25])


    # 컨볼루션 레이어 이후의 다운샘플링을 고려하여 입력 길이 조정
    # 풀링 레이어로 인해 시간 축 길이가 감소하므로 전체 다운샘플링 팩터를 계산합니다.
    total_downsample_factor = 5
    adjusted_input_length = inputs.shape[-2] // total_downsample_factor
    input_lengths = torch.full((len(batch),), adjusted_input_length, dtype=torch.int32)

    # 라벨을 패딩합니다.
    #labels = pad_sequence(labels, batch_first=True, padding_value=0)
    # 라벨을 하나의 텐서로 병합합니다.
    labels = torch.cat(labels)

    return inputs, labels, input_lengths, label_lengths

class Dataset(Dataset):
    def __init__(self, mel_specs, labels_integer, max_label_length):
        self.X_input_mels = mel_specs # (오디오 수, 300, 45, 1)
        self.Y_labels = labels_integer
        self.Z_max_label_length = max_label_length
        self.n_samples = len(self.X_input_mels)
        self.indexes = np.arange(self.n_samples)

    def __len__ (self):
        return self.n_samples

    def __getitem__(self, idx):
        X_input_mel = torch.tensor(self.X_input_mels[idx])
        Y_label = torch.tensor(self.Y_labels[idx])

        input_length = torch.tensor([X_input_mel.shape[0]], dtype=torch.int32) # (width, height, channels)
        label_length = torch.tensor([self.Z_max_label_length], dtype=torch.int32)

        return X_input_mel, Y_label, input_length, label_length

class ATCLoad() :
    def __init__(self, audio_folder_path, csv_path, t_width, f_mels, CMUdict, downsample_factor):
        self.audio_folder_path = audio_folder_path
        self.csv_path = csv_path
        self.t_width = t_width
        self.f_mels = f_mels
        self.CMUdict = CMUdict
        self.downsample_factor = downsample_factor
        self.df = pd.read_csv(csv_path)

        self.X_mel_specs = []
        self.Z_input_lengths = []
        self.Y_labels_integer = []
        self.Z_labels_lengths = []
        self.Z_max_label_length = 1
        self.num_classes = len(CMUdict)+1

        self.load_data()

    def text_transform(self, text_line) :
        phoneme_to_idx = {p: i+1 for i, p in enumerate(self.CMUdict)}  # Start indices from 1 (0 is reserved for blank)
        idx_to_phoneme = {i+1: p for i, p in enumerate(self.CMUdict)}
        phoneme_to_idx['-'] = 0
        idx_to_phoneme[0] = '-'

        phoneme_seq = text_line.split(" ")
        converted_line = []
        for item in phoneme_seq:
            idx = phoneme_to_idx[item]
            converted_line.append(idx)
        
        return converted_line

    def load_data(self) :
        # Load Origin Labels
        df = pd.read_csv(self.csv_path)
        Y_labels_origin = df['Phonemes']        
        print("[ATCLoader]", len(Y_labels_origin),  "Labels Loaded!")

        # Convert to Integer Labels & Get Label Lengths
        labels_integer = []
        labels_lengths = []
        for phoneme_seq in Y_labels_origin:
            line = self.text_transform(phoneme_seq)
            labels_integer.append(line)           ## INTEGER LABELS
            labels_lengths.append(len(line))      ## LABELS LENGTH

        # labels_integer = nn.utils.rnn.pad_sequence(labels_integer, batch_first=True)
        self.Y_labels_integer = labels_integer    # return > class instance (Y_labels_integer)
        self.Z_labels_lengths = labels_lengths    # return > class instance (Z_labels_lengths)
        self.Z_max_label_length = max(labels_lengths)
        print("[ATCLoader]", len(self.Y_labels_integer),  "Labels Preprossed!")
        
        # Load AudioFiles with MEL Conversion
        mel_specs = []
        input_lengths = []

        print("[ATCLoader]", len(df['AudioFile']), "Audio Loaded!")
        for i in df['AudioFile']:
            audio_file = os.path.join(self.audio_folder_path, i + ".mp3")
            sig, sr = librosa.load(audio_file, sr=16000)
            mel_spec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=1024, hop_length=256, n_mels=self.f_mels) # (height, width)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = np.expand_dims(mel_spec.T, axis=-1)
            # print(os.path.basename(audio_file), "mel_spec_db.shape : ", mel_spec.shape)

            current_width = mel_spec.shape[0]

            if current_width < self.t_width:
                pad_width = self.t_width - current_width
                left_pad = pad_width // 2
                right_pad = pad_width - left_pad

                mel_spec = np.pad(
                    mel_spec,
                    pad_width=((left_pad, right_pad), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=0
                )

            elif current_width > self.t_width:
                scale_factor = self.t_width / current_width
                mel_spec = zoom(mel_spec, (scale_factor, 1, 1), order=1)

            mel_specs.append(mel_spec)
            input_lengths.append(mel_spec.shape[0]//self.downsample_factor)
        

        self.X_mel_specs = mel_specs          # return > class instance (X_mel_specs)
        self.Z_input_lengths = input_lengths  # return > class instance (Z_input_lengths)
        print("[ATCLoader]", len(self.X_mel_specs), "Audio-Mel Preprocssed!")

        return self.X_mel_specs, self.Y_labels_integer, self.Z_input_lengths, self.Z_labels_lengths

    def get_dataloader(self, batch_size):
        dataset = Dataset(self.X_mel_specs, self.Y_labels_integer, self.Z_labels_lengths)
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.9) # 90%
        validation_size = dataset_size - train_size

        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        return train_loader, validation_loader
    
    def get_max_label_length(self):
        return self.Z_max_label_length


class LibriLoad() :
    def __init__(self, t_width, f_mels, CMUdict, downsample_factor, data_dir="Processed_LibriSpeech", device=None):
        self.libri_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)
        self.data_dir = data_dir
        self.t_width = t_width
        self.f_mels = f_mels
        self.CMUdict = CMUdict
        self.downsample_factor = downsample_factor

        self.X_mel_specs = []
        self.Z_input_lengths = []
        self.Y_labels_integer = []
        self.Z_labels_lengths = []
        self.Z_max_label_length = 1
        self.num_classes = len(CMUdict)+1

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # 데이터 폴더 확인
        os.makedirs(data_dir, exist_ok=True)

        # 로드 또는 변환
        if self.check_processed_data():
            self.load_processed_data()
        else:
            self.load_data()
            self.save_processed_data()

    def check_processed_data(self):
        """저장된 데이터가 있는지 확인"""
        required_files = ["X_mel_specs.npy", "Z_input_lengths.npy", "Y_labels_integer.pkl", "Z_labels_lengths.npy"]
        return all(os.path.exists(os.path.join(self.data_dir, file)) for file in required_files)

    def save_processed_data(self):
        """변환된 데이터를 로컬에 저장"""
        print("[SAVING DATA] Saving processed data...")
        np.save(os.path.join(self.data_dir, "X_mel_specs.npy"), self.X_mel_specs)
        np.save(os.path.join(self.data_dir, "Z_input_lengths.npy"), self.Z_input_lengths)
        with open(os.path.join(self.data_dir, "Y_labels_integer.pkl"), "wb") as f:
            pickle.dump(self.Y_labels_integer, f)
        np.save(os.path.join(self.data_dir, "Z_labels_lengths.npy"), self.Z_labels_lengths)
        print("[SAVING DATA] Processed data saved successfully!")

    def load_processed_data(self):
        """저장된 데이터를 로드"""
        print("[LOADING DATA] Loading processed data...")
        self.X_mel_specs = np.load(os.path.join(self.data_dir, "X_mel_specs.npy"), allow_pickle=True)
        self.Z_input_lengths = np.load(os.path.join(self.data_dir, "Z_input_lengths.npy"), allow_pickle=True)
        with open(os.path.join(self.data_dir, "Y_labels_integer.pkl"), "rb") as f:
            self.Y_labels_integer = pickle.load(f)
        self.Z_labels_lengths = np.load(os.path.join(self.data_dir, "Z_labels_lengths.npy"), allow_pickle=True)
        self.Z_max_label_length = max(self.Z_labels_lengths)
        print("[LOADING DATA] Processed data loaded successfully!")

    def text_transform(self, text_line) :
        g2p = G2p()

        def remove_stress(ph):
            return [re.sub(r'[012]', '', phoneme) for phoneme in ph]

        phoneme_to_idx = {p: i+1 for i, p in enumerate(self.CMUdict)}  # Start indices from 1 (0 is reserved for blank)
        idx_to_phoneme = {i+1: p for i, p in enumerate(self.CMUdict)}
        phoneme_to_idx['-'] = 0
        idx_to_phoneme[0] = '-'

        # print(text_line)
        words = text_line.split()    
        phonemes_line = []
        for word in words:
            phonemes = g2p(word)
            phonemes = remove_stress(phonemes)
            for phoneme in phonemes:
                phonemes_line.append(phoneme)
        phoneme_seq = phonemes_line
        # print(phoneme_seq)
        converted_line = []
        for item in phoneme_seq:
            idx = phoneme_to_idx[item]
            converted_line.append(idx)
        return converted_line
    
    def load_data(self) :
        mel_specs = []
        input_lengths = []
        labels_integer = []
        labels_lengths = []

        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=self.f_mels
        ).to(self.device)

        for (waveform, _, utterance, _, _, _) in self.libri_dataset:
            waveform = waveform.mean(dim=0).to(self.device)
            mel_spec = mel_transform(waveform)  # Apply MelSpectrogram transformation
            mel_spec = torchaudio.functional.amplitude_to_DB(
                mel_spec,
                multiplier=10.0,
                amin=1e-10,
                db_multiplier=torch.log10(torch.maximum(mel_spec.max(), torch.tensor(1e-10, device=self.device)))
            )

            mel_spec = mel_spec.T.unsqueeze(-1)
            mel_spec = mel_spec.cpu().numpy()  # Transpose and add channel dimension
            print("mel_spec.shape : ", mel_spec.shape)

            current_width = mel_spec.shape[0]

            if current_width < self.t_width:
                pad_width = self.t_width - current_width
                left_pad = pad_width // 2
                right_pad = pad_width - left_pad

                mel_spec = np.pad(
                    mel_spec,
                    pad_width=((left_pad, right_pad), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            
            elif current_width > self.t_width:
                scale_factor = self.t_width / current_width
                mel_spec = zoom(mel_spec, (scale_factor, 1, 1), order=1)


            mel_specs.append(mel_spec)
            input_lengths.append(mel_spec.shape[0]//self.downsample_factor)
            
            labels_integer.append(self.text_transform(utterance.lower()))
            labels_lengths.append(len(self.text_transform(utterance.lower())))
            
        self.X_mel_specs = mel_specs
        self.Z_input_lengths = input_lengths
        self.Y_labels_integer = labels_integer
        self.Z_labels_lengths = labels_lengths
        self.Z_max_label_length = max(labels_lengths)

        print("[LIBRISPEECH]", len(labels_integer),  "Labels Preprossed!")
        print("[LIBRISPEECH]", len(mel_specs), "Audio-Mel Preprossed!")

        return self.X_mel_specs, self.Y_labels_integer, self.Z_input_lengths, self.Z_labels_lengths

    def get_dataloader(self, batch_size):
        dataset = Dataset(self.X_mel_specs, self.Y_labels_integer, self.Z_labels_lengths)
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.9) # 90
        validation_size = dataset_size - train_size

        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        return train_loader, validation_loader

        

