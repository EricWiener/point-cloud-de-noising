import torch.nn as nn
import torch.optim as optim
from model.weathernet import WeatherNet
from pcd_dataset import PCDDataset
from torch.utils.data import DataLoader

# DATASET_PATH = "/Users/ericwiener/repositories/point-cloud-de-noising/data"
DATASET_PATH = (
    "/home/elliot/Desktop/cnn_denoising_dataset/train"  # TODO: use train_road?
)


def main():
    dataset = PCDDataset(DATASET_PATH, recursive=True)
    print(f"Found {len(dataset)} files")

    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    net = WeatherNet(num_classes=4)  # No label (0), clear (1), rain (2), fog (3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), eps=1e-8)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (distance, reflectivity, labels) in enumerate(loader):

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
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

    print("Finished training")


if __name__ == "__main__":
    main()
