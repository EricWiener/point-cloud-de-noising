import torch

from pcd_de_noising.model.lilanet import LiLaNet


def test_lilanet_size_check():
    num_classes, height, width = 4, 64, 512

    model = LiLaNet(num_classes)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])
