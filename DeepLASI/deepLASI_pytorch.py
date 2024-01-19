import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='relu', use_bias=True):
        super(Conv1DBN, self).__init__()
        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        return x

def OS_CNN_Block(branch_0, block_filters, max_prime, pool=False, res=None, activation='relu'):
    branches = [Conv1DBN(branch_0.size(1), block_filters, 1)]
    for number in range(2, max_prime+1):
        if all(number % i != 0 for i in range(2, number)):
            branches.append(Conv1DBN(branch_0.size(1), block_filters, number))

    m = torch.cat([branch(branch_0) for branch in branches], dim=1)
    m = nn.BatchNorm1d(m.size(1))(m)
    if res is not None:
        res = res + m
        x = F.relu(res) if activation == 'relu' else res
    else:
        x = F.relu(m) if activation == 'relu' else m
    return x

def Final_CNN(branch_0, block_filters, res=None, activation='relu'):
    branches = [Conv1DBN(branch_0.shape[1], block_filters, 1), Conv1DBN(branch_0.shape[1], block_filters, 2)]
    m = torch.cat([branch(branch_0) for branch in branches], dim=1)
    m = nn.BatchNorm1d(m.shape[1])(m)
    if res is not None:
        m = m + res
        x = F.relu(m) if activation == 'relu' else m
    else:
        x = F.relu(m) if activation == 'relu' else m
    return x

class build_model(nn.Module):
    def __init__(self, channels=6, classes=12, model_type="trace_classifier", max_prime=23, resnet=False):
        super(build_model, self).__init__()
        n_filters = 32 if model_type == "trace_classifier" else 64

        # Initial block
        self.initial_block = OS_CNN_Block(torch.randn(1, channels, 100), n_filters, max_prime, pool=False)  # Example input

        # Additional blocks
        self.blocks = nn.ModuleList()
        if resnet:  # includes residual connections (only recommended for trace classifiers)
            for _ in range(2):
                sc = self.initial_block
                block1 = OS_CNN_Block(torch.randn(1, sc.shape[1], 100), n_filters, max_prime, pool=True)
                block2 = OS_CNN_Block(block1, n_filters, max_prime, pool=True, res=sc)
                self.blocks.extend([block1, block2])
            self.dropout1 = nn.Dropout(0.2)
            self.lstm1 = nn.LSTM(input_size=block2.shape[1], hidden_size=128, batch_first=True, bidirectional=True)
            self.dropout2 = nn.Dropout(0.5)
            self.lstm2 = nn.LSTM(input_size=256, hidden_size=32, batch_first=True, bidirectional=True)  # 128 * 2 due to bidirectional
        elif model_type == "trace_classifier":
            self.block = OS_CNN_Block(self.initial_block, n_filters, max_prime, pool=True)
            self.final_cnn = Final_CNN(self.block, n_filters)
            self.dropout1 = nn.Dropout(0.1)
            self.lstm1 = nn.LSTM(input_size=self.final_cnn.shape[1], hidden_size=128, batch_first=True, bidirectional=True)
            self.dropout2 = nn.Dropout(0.5)
            self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)  # 128 * 2 due to bidirectional
        else:
            # For state classifier or number of states classifier
            self.final_cnn = Final_CNN(self.initial_block, 32)
            self.dropout1 = nn.Dropout(0.5)
            self.lstm1 = nn.LSTM(input_size=self.final_cnn.shape[1], hidden_size=128, batch_first=True, bidirectional=True)
            self.dropout2 = nn.Dropout(0.5)
            self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)  # 128 * 2 due to bidirectional

        self.dropout3 = nn.Dropout(0.5)
        self.fc = nn.Linear(256, classes)  # 128 * 2 due to bidirectional LSTM

    def forward(self, x):
        x = self.initial_block(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_cnn(x) if hasattr(self, 'final_cnn') else x
        x = self.dropout1(x)
        x, _ = self.lstm1(x)
        x = self.dropout2(x)
        x, _ = self.lstm2(x)
        x = self.dropout3(x)
        x = self.fc(x)
        return x



model = build_model(channels=2, classes=2, model_type="state_classifier")









