import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F
from .lilanet import LiLaBlock


class WeatherNet(nn.Module):
    """
    Implements WeatherNet model from
    `"CNN-based Lidar Point Cloud De-Noising in Adverse Weather"
    <https://arxiv.org/abs/1912.03874>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, num_classes=3):
        super(WeatherNet, self).__init__()

        self.lila1 = LiLaBlock(2, 96, modified=True)
        self.lila2 = LiLaBlock(96, 128, modified=True)
        self.lila3 = LiLaBlock(128, 256, modified=True)
        self.lila4 = LiLaBlock(256, 256, modified=True)
        self.dropout = nn.Dropout2d()
        self.lila5 = LiLaBlock(256, 128, modified=True)
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
        """Forward pass of Weathernet

        Args:
            distance (torch.tensor): tensor of shape [B x 32 x 400 x 1]
            reflectivity (torch.tensor): tensor of shape [B x 32 x 400 x 1]]

        Returns:
            torch.tensor: predictions of shape [B x num_classes x 32 x 400]
        """
        x = torch.cat([distance, reflectivity], 1)
        x = self.lila1(x)
        x = self.lila2(x)
        x = self.lila3(x)
        x = self.lila4(x)
        x = self.dropout(x)
        x = self.lila5(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":
    num_classes, height, width = 3, 64, 512

    model = WeatherNet(num_classes)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])

    print("Pass size check.")