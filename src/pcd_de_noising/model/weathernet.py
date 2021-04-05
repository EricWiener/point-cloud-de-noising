import pytorch_lightning as pl
import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1, HammingDistance

from .lilanet import LiLaBlock


class WeatherNet(pl.LightningModule):
    """
    Implements WeatherNet model from
    `"CNN-based Lidar Point Cloud De-Noising in Adverse Weather"
    <https://arxiv.org/abs/1912.03874>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.lila1 = LiLaBlock(2, 96, modified=True)
        self.lila2 = LiLaBlock(96, 128, modified=True)
        self.lila3 = LiLaBlock(128, 256, modified=True)
        self.lila4 = LiLaBlock(256, 256, modified=True)
        self.dropout = nn.Dropout2d()
        self.lila5 = LiLaBlock(256, 128, modified=True)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

        # Metrics
        # self.train_f1 = F1(num_classes)
        # self.train_hamming = HammingDistance()
        # self.val_f1 = F1(num_classes)
        # self.val_hamming = HammingDistance()

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

    def predict(self, distance, reflectivity):
        """Run inference

        Args:
            distance (torch.tensor): distance tensor of dim [B, 1, 32, 400]
            reflectivity (torch.tensor): reflectivity tensor of dim [B, 1, 32, 400]

        Returns:
            torch.tensor: predictions of dim [B, 32, 400]
        """

        with torch.no_grad():
            # logits is [B, 4 (num classes), 32, 400]
            logits = self(distance, reflectivity)

            # predictions is [B, 32, 400]
            predictions = torch.argmax(logits, dim=1)

        return predictions

    def shared_step(self, distance, reflectivity, labels):
        # logits is [B, 4 (num classes), 32, 400]
        logits = self(distance, reflectivity)

        # labels is [B, 32, 400]
        loss = F.cross_entropy(logits, labels)

        # predictions is [B, 32, 400]
        predictions = torch.argmax(logits, dim=1)

        return loss, logits, predictions

    def training_step(self, batch, batch_idx):
        distance, reflectivity, labels = batch
        loss, logits, predictions = self.shared_step(distance, reflectivity, labels)

        self.log("train_loss", loss)

        # Log Metrics
        # self.train_f1(predictions, labels)
        # self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        # self.train_hamming(logits, labels)
        # self.log("train_hamming", self.train_hamming, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        distance, reflectivity, labels = batch
        loss, logits, predictions = self.shared_step(distance, reflectivity, labels)

        self.log("val_loss", loss)

        # self.val_f1(predictions, labels)
        # self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        # self.val_hamming(logits, labels)
        # self.log("val_hamming", self.val_hamming, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), eps=1e-8)
        return optimizer


if __name__ == "__main__":
    num_classes, height, width = 3, 64, 512

    model = WeatherNet(num_classes)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])

    print("Pass size check.")
