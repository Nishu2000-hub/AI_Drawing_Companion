import torch.nn as nn

class SketchLSTM(nn.Module):
    def __init__(self, input_dim: int = 3,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 num_classes: int = 50):
       
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        
        last_hidden = h_n[-1] 
        logits = self.classifier(last_hidden)
        return logits
