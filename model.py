import torch
import torch.nn as nn
import torch.nn.functional as F


def clipped_relu(x, max_value=6):
    """Clipped ReLU implementation."""
    return torch.clamp(F.relu(x), max=max_value)

class BassCRNN11(nn.Module):
    def __init__(self, num_classes):
        super(BassCRNN11, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 512, kernel_size=(5, 8), padding=(2, 4))
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn2 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn7 = nn.BatchNorm2d(512)

        # Linear transformation for RNN input
        self.dense = nn.Linear(21504, 128)  # Adjusted to match flattened size after pooling (= 512 * 71)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.bn_lstm1 = nn.BatchNorm1d(512)

        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.bn_lstm2 = nn.BatchNorm1d(512)

        # Final output layer
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv layers with ReLU and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 1))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 1))

        x = F.relu(self.bn7(self.conv7(x)))

        # 출력 (Batch_size, Channels, Width, Height) = torch.Size([8, 512, 71, 24])

        # Reshape for RNN
        sizes = x.size()
        x = x.reshape(sizes[0], sizes[1] * sizes[3], sizes[2])
        x = x.transpose(1, 2)
        # RNN Input : [(batch_size, seq_length, input_features)]

        # Fully connected layer
        x = F.relu(self.dense(x))

        # LSTM layers(
        x, _ = self.lstm1(x)
        x = self.bn_lstm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, _ = self.lstm2(x)
        x = self.bn_lstm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Output layer with softmax
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y_pred = F.log_softmax(x, dim=-1)

        return y_pred  # torch.Size([8, 71, 40])


class BassCRNN12(nn.Module):
    def __init__(self, num_classes, max_label_length):
        super(BassCRNN12, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 512, kernel_size=(5, 8), padding=(2, 4))
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn2 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn7 = nn.BatchNorm2d(512)

        # Linear transformation for RNN input
        self.dense = nn.Linear(21504, 128)  # Adjusted to match flattened size after pooling (= 512 * 71)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.bn_lstm1 = nn.BatchNorm1d(512)

        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.bn_lstm2 = nn.BatchNorm1d(512)

        # Final output layer
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv layers with Clipped ReLU and pooling
        x = clipped_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 1))

        x = clipped_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 1))

        x = clipped_relu(self.bn7(self.conv7(x)))

        # Reshape for RNN
        sizes = x.size()
        x = x.reshape(sizes[0], sizes[1] * sizes[3], sizes[2])
        x = x.transpose(1, 2)
        # RNN Input : [(batch_size, seq_length, input_features)]

        # Fully connected layer
        x = clipped_relu(self.dense(x))

        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.bn_lstm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, _ = self.lstm2(x)
        x = self.bn_lstm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Output layer with softmax
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y_pred = F.log_softmax(x, dim=-1)

        return y_pred  # torch.Size([8, 71, 40])

class BassCRNN21(nn.Module):
    def __init__(self, num_classes, max_label_length):
        super(BassCRNN21, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 512, kernel_size=(5, 8), padding=(2, 4))
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn2 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn7 = nn.BatchNorm2d(512)

        # Linear transformation for RNN input
        self.dense = nn.Linear(21504, 128)  # Adjusted to match flattened size after pooling (= 512 * 71)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.bn_lstm1 = nn.BatchNorm1d(512)

        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.bn_lstm2 = nn.BatchNorm1d(512)

        # Final output layer
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv layers with ReLU and pooling
        #print("input.sizen : ", x.size())

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 1))
        #print("conv1_x.sizen : ", x.size())

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 1))
        #print("conv2_x.sizen : ", x.size())

        x = F.relu(self.bn7(self.conv7(x)))
        #print("conv7_x.sizen : ", x.size())

        # 출력 (Batch_size, Channels, Width, Height) = torch.Size([8, 512, 71, 24])

        # Reshape for RNN
        sizes = x.size()
        x = x.reshape(sizes[0], sizes[1] * sizes[3], sizes[2])
        x = x.transpose(1, 2)
        # RNN Input : [(batch_size, seq_length, input_features)]

        # Fully connected layer
        x = F.relu(self.dense(x))

        # LSTM layers(
        x, _ = self.lstm1(x)
        x = self.bn_lstm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, _ = self.lstm2(x)
        x = self.bn_lstm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Output layer with softmax
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y_pred = F.log_softmax(x, dim=-1)

        return y_pred  # torch.Size([8, 71, 40])

class BassCRNN31(nn.Module):
    def __init__(self, num_classes, max_label_length):
        super(BassCRNN31, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 512, kernel_size=(5, 8), padding=(2, 4))
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn2 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(5,1), padding=(2, 4))
        self.bn7 = nn.BatchNorm2d(512)

        # Linear transformation for RNN input
        self.dense = nn.Linear(21504, 128)  # Adjusted to match flattened size after pooling (= 512 * 71)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.bn_lstm1 = nn.BatchNorm1d(512)

        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.bn_lstm2 = nn.BatchNorm1d(512)

        # Final output layer
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv layers with ReLU and pooling
        #print("input.sizen : ", x.size())

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 1))
        #print("conv1_x.sizen : ", x.size())

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 1))
        #print("conv2_x.sizen : ", x.size())

        x = F.relu(self.bn7(self.conv7(x)))
        #print("conv7_x.sizen : ", x.size())

        # 출력 (Batch_size, Channels, Width, Height) = torch.Size([8, 512, 71, 24])

        # Reshape for RNN
        sizes = x.size()
        x = x.reshape(sizes[0], sizes[1] * sizes[3], sizes[2])
        x = x.transpose(1, 2)
        # RNN Input : [(batch_size, seq_length, input_features)]

        # Fully connected layer
        x = F.relu(self.dense(x))

        # LSTM layers(
        x, _ = self.lstm1(x)
        x = self.bn_lstm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, _ = self.lstm2(x)
        x = self.bn_lstm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Output layer with softmax
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y_pred = F.log_softmax(x, dim=-1)

        return y_pred  # torch.Size([8, 71, 40])



class BaseModel():
    def __init__(self, num_classes, model_code):
        if(model_code == "01_1"):
            self.model = BassCRNN11(num_classes)
        elif(model_code == "01_2"):
            self.model = BassCRNN12(num_classes)
        elif(model_code == "02_1"):
            self.model = BassCRNN21(num_classes)
        elif(model_code == "03_1"):
            self.model = BassCRNN31(num_classes)
        elif(model_code == "04_1"):
            self.model = BassCRNN11(num_classes)
        elif(model_code == "04_2"):
            self.model = BassCRNN11(num_classes)
        else:
            self.model = BassCRNN11(num_classes)

    def get_model(self):
        return self.model
    
    def get_loss_fn(self):
        return nn.CTCLoss(blank=0)
    
    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    