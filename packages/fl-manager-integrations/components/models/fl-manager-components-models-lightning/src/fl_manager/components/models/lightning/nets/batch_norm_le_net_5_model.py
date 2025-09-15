from torch import nn

from fl_manager.components.models.lightning.nets._mnist_net import BaseMNISTNet


class BatchNormLeNet5Model(BaseMNISTNet):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)
