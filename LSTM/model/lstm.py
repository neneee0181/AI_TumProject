import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]     # 마지막 타임스텝
        out = self.fc(out)
        return out
