import torch
from torch.utils.data import Dataset, DataLoader
import os

class VideoDataset(Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.files = [f for f in os.listdir(features_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.features_dir, self.files[idx])
        features = torch.load(feature_path)  # [T, 512]
        return features, self.files[idx]  # return name for debugging

if __name__ == "__main__":
    dataset = VideoDataset("../data/features")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch_features, batch_name in dataloader:
        print(batch_name, batch_features.shape)

