from pcd_dataset import PCDDataset
from torch.utils.data import DataLoader

DATASET_PATH = "/home/elliot/Desktop/cnn_denoising_dataset/train"  # TODO: use train_road?

def main():
    dataset = PCDDataset(DATASET_PATH, recursive=True)
    print(f"Found {len(dataset)} files")

    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)

    for data, label in loader:
        print(data, label)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
