import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model.weathernet import WeatherNet
from pcd_dataset import PCDDataset
from torch.utils.data import DataLoader

# DATASET_PATH = "/Users/ericwiener/repositories/point-cloud-de-noising/data"
DATASET_PATH = (
    "/home/elliot/Desktop/cnn_denoising_dataset/train"  # TODO: use train_road?
)


def train(num_epochs, cuda):

    use_cuda = cuda and torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    dataset = PCDDataset(DATASET_PATH, recursive=True)
    print(f"Found {len(dataset)} files")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    net = WeatherNet(num_classes=4)  # No label (0), clear (1), rain (2), fog (3)
    if use_cuda: net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), eps=1e-8)

    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, (distance, reflectivity, labels) in enumerate(loader):

            if use_cuda:
                distance = distance.cuda()
                reflectivity = reflectivity.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs is [B, 4 (num classes), 32, 400]
            outputs = net(distance, reflectivity)

            # Labels is [B, 32, 400]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 0:  # print every 2 mini-batches
                print(f"[{epoch + 1}/{num_epochs}, {(i + 1):5}/{len(loader)}]",
                      f"loss: {(running_loss / 2):.3f}"
                )
                running_loss = 0.0

    print("Finished training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA (default: false)')
    args = parser.parse_args()
    train(num_epochs=args.epochs, cuda=args.cuda)
