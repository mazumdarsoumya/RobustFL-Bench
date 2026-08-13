import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    def __init__(self, in_channels=1, image_size=28, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * image_size * image_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _replace_conv(module, in_channels):
    conv = module
    return nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
    )


def _resnet18(in_channels, num_classes):
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _mobilenet_v3_small(in_channels, num_classes):
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None, num_classes=num_classes)
    model.features[0][0] = _replace_conv(model.features[0][0], in_channels)
    return model


def _shufflenet_v2(in_channels, num_classes):
    from torchvision.models import shufflenet_v2_x1_0

    model = shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
    model.conv1[0] = _replace_conv(model.conv1[0], in_channels)
    return model


def _efficientnet_b0(in_channels, num_classes):
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None, num_classes=num_classes)
    model.features[0][0] = _replace_conv(model.features[0][0], in_channels)
    return model


def get_model(name: str, dataset_name: str, num_classes: int, image_size: int | None = None):
    name = name.lower()
    in_channels = 1 if dataset_name.lower() in {"mnist", "fashionmnist"} else 3
    size = int(image_size or (28 if in_channels == 1 else 32))

    if name == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "mlp":
        return MLP(in_channels=in_channels, image_size=size, num_classes=num_classes)
    if name == "resnet18":
        return _resnet18(in_channels, num_classes)
    if name == "mobilenet_v3_small":
        return _mobilenet_v3_small(in_channels, num_classes)
    if name in {"shufflenet_v2", "shufflenet_v2_x1_0"}:
        return _shufflenet_v2(in_channels, num_classes)
    if name == "efficientnet_b0":
        return _efficientnet_b0(in_channels, num_classes)
    raise ValueError(f"Unsupported model: {name}")
