from torch import nn

from fl_manager.components.models.lightning.nets._mnist_net import BaseMNISTNet


def _depth_wise_separable_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class MNISTMobileNetModel(BaseMNISTNet):
    def __init__(
        self, num_classes: int, in_channels: int, learning_rate=1e-3, weight_decay=1e-4
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self._in_channels = in_channels
        self.model = nn.Sequential(
            nn.Conv2d(self._in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            _depth_wise_separable_conv(32, 64),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            _depth_wise_separable_conv(64, 128),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            _depth_wise_separable_conv(128, 128),
            nn.AdaptiveAvgPool2d(1),  # Output: (batch, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self._num_classes),
        )

    def forward(self, x):
        return self.model(x)
