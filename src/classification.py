import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === Data Loading and Preprocessing ===
def load_point_cloud(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return np.load(file_path)

def point_cloud_to_voxel(points, grid_size=32):
    voxel_grid = np.zeros((grid_size, grid_size, grid_size))
    normalized_points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    voxel_indices = (normalized_points * (grid_size - 1)).astype(int)
    for idx in voxel_indices:
        voxel_grid[tuple(idx)] = 1
    assert voxel_grid.shape == (grid_size, grid_size, grid_size), "Voxel grid shape mismatch!"
    return voxel_grid

# === Visualize Voxel Grid ===
def visualize_voxel(voxel_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.nonzero(voxel_grid)
    ax.scatter(x, y, z, zdir='z', c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# === Dataset Class for Multiple Datasets ===
class MultiDataset(Dataset):
    def __init__(self, files, labels, grid_size=32):
        self.files = files
        self.labels = labels
        self.grid_size = grid_size

        if len(self.files) != len(self.labels):
            raise ValueError("The number of labels must match the number of files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        point_cloud = load_point_cloud(file_path)
        voxel_grid = point_cloud_to_voxel(point_cloud, self.grid_size)
        label = self.labels[idx]
        return voxel_grid[np.newaxis, :, :, :], label

# === ResNet3D and Training Logic ===
class ResNet3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def validate_model(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.float(), targets.long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.long()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss, val_accuracy = validate_model(model, val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.float(), targets.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# === Main Execution ===
if __name__ == "__main__":
    print("Running...")
    # Base directory containing datasets
    base_dir = "./datasets"
    sub_dirs = ["1", "2", "3", "4", "5"]  # Dataset folder names

    # Combine all files and labels
    all_files = []
    all_labels = []
    num_classes = 4  # Adjust based on your classification task
    for i, sub_dir in enumerate(sub_dirs):
        dataset_dir = os.path.join(base_dir, sub_dir, "npy")
        files = sorted(os.listdir(dataset_dir))
        labels = [i % num_classes for i in range(len(files))]  # Assign labels cyclically
        all_files.extend([os.path.join(dataset_dir, file) for file in files])
        all_labels.extend(labels)

    print(npy_data)

    # # Train/test split
    # train_files, test_files, train_labels, test_labels = train_test_split(
    #     all_files, all_labels, test_size=0.2, random_state=42
    # )

    # # Create datasets and DataLoaders
    # train_dataset = MultiDataset(train_files, train_labels, grid_size=32)
    # test_dataset = MultiDataset(test_files, test_labels, grid_size=32)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # # Visualize a sample voxel grid
    # sample_voxel, _ = train_dataset[0]
    # visualize_voxel(sample_voxel.squeeze(0))

    # # Initialize and train model
    # model = ResNet3D(ResNet3DBlock, [2, 2, 2, 2], num_classes)
    # train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001)

    # # Evaluate the model
    # evaluate_model(model, test_loader)

    # # Confirm Completion
    # print("Done Running")