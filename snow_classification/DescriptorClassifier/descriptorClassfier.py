from torch import nn

class DescriptorClassifier(nn.Module):
    def __init__(self, input_size):
        super(DescriptorClassifier, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 196),
            nn.ReLU(),
            nn.Linear(196, 196),
            nn.ReLU(),
            nn.Linear(196, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x