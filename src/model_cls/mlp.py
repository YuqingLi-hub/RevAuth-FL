import torch.nn as nn

class MLP(nn.Module):
    # _784_126_728_10
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, 126, bias=True)
        self.fc2 = nn.Linear(126, 728, bias=True)
        self.fc3 = nn.Linear(728, num_classes, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, 1, 28, 28) or (B, 784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)  # flatten
        elif x.dim() == 3:  # (B, 28, 28)
            x = x.view(x.size(0), -1)
        # forward
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # logits
        return x