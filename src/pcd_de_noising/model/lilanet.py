import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F
from .padding import conv2d_get_padding

pretrained_models = {
    "kitti": {
        "url": "https://github.com/TheCodez/pytorch-LiLaNet/releases/download/0.1/lilanet_45.5-75c06618.pth",
        "num_classes": 4,
    }
}


def lilanet(pretrained=None, num_classes=13):
    """Constructs a LiLaNet model.

    Args:
        pretrained (string): If not ``None``, returns a pre-trained model. Possible values: ``kitti``.
        num_classes (int): number of output classes. Automatically set to the correct number of classes
            if ``pretrained`` is specified.
    """
    if pretrained is not None:
        model = LiLaNet(pretrained_models[pretrained]["num_classes"])
        model.load_state_dict(
            hub.load_state_dict_from_url(pretrained_models[pretrained]["url"])
        )
        return model

    model = LiLaNet(num_classes)
    return model


class LiLaNet(nn.Module):
    """
    Implements LiLaNet model from
    `"Boosting LiDAR-based Semantic Labeling by Cross-Modal Training Data Generation"
    <https://arxiv.org/abs/1804.09915>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, num_classes=13):
        super(LiLaNet, self).__init__()

        self.lila1 = LiLaBlock(2, 96)
        self.lila2 = LiLaBlock(96, 128, modified=True)
        self.lila3 = LiLaBlock(128, 256)
        self.lila4 = LiLaBlock(256, 256)
        self.lila5 = LiLaBlock(256, 128)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, distance, reflectivity):
        x = torch.cat([distance, reflectivity], 1)
        x = self.lila1(x)
        x = self.lila2(x)
        x = self.lila3(x)
        x = self.lila4(x)
        x = self.lila5(x)

        x = self.classifier(x)

        return x


class LiLaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n,
        modified=False,
    ):
        super(LiLaBlock, self).__init__()
        self.modified = modified
        self.branch1 = BasicConv2d(in_channels, n, kernel_size=(7, 3), padding=(2, 0))
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3)
        self.branch3 = BasicConv2d(in_channels, n, kernel_size=(3, 7), padding=(0, 2))

        if modified:
            self.pad_input = None
            self.branch4 = BasicConv2d(
                self.branch1.in_channels,
                self.branch1.out_channels,
                kernel_size=3,
                dilation=2,
            )

        number_branches = 4 if modified else 3
        self.conv = BasicConv2d(n * number_branches, n, kernel_size=1, padding=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        if self.modified and self.pad_input is None:
            # If we are using the modified LilaBlock with dilation,
            # then we need to calculate padding
            # We calculate the padding to use during the initial pass
            # and save it for future runs
            self.pad_input = nn.ZeroPad2d(
                conv2d_get_padding(
                    x.shape[-2::],
                    branch1.shape[-2::],
                    kernel_size=3,
                    stride=1,
                    dilation=2,
                )
            )

        if self.modified:
            padded_x = self.pad_input(x)
            branch4 = self.branch4(padded_x)
            output = torch.cat([branch1, branch2, branch3, branch4], 1)
        else:
            output = torch.cat([branch1, branch2, branch3], 1)

        output = self.conv(output)

        return output


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == "__main__":
    num_classes, height, width = 4, 64, 512

    model = LiLaNet(num_classes)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])

    print("Pass size check.")
