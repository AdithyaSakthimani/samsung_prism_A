import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VideoDataset

# Hyperparameters
input_dim = 512
hidden_dim = 256
num_classes = 10  # number of event classes
batch_size = 1
epochs = 5
lr = 1e-3

# Dataset and DataLoader
dataset = VideoDataset("../data/features")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Simple classifier
class BaselineClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, T, 512]
        x = x.mean(dim=1)  # average over time
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BaselineClassifier(input_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Dummy labels for demonstration (replace with real labels)
for epoch in range(epochs):
    for features, names in dataloader:
        features = features.to(device)  # [batch, T, 512]
        labels = torch.randint(0, num_classes, (features.size(0),)).to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "../training/baseline.pt")
print("Model saved as baseline.pt")

