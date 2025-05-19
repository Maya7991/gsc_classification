import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import snntorch.spikeplot as splt
import os
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TTFS encoding parameters
num_steps = 100

def ttfs_encode(x):
    return spikegen.latency(x, num_steps=num_steps, normalize=True, linear=True)

# MelSpectrogram Transform
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=40
)

# Dataset Setup
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [os.path.join(self._path, line.strip()) for line in f]
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

labels = sorted(list(set(dat[2] for dat in SubsetSC("training"))))
label_to_index = {label: i for i, label in enumerate(labels)}

# Collate function for DataLoader
def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        mel = transform(waveform).squeeze(0)  # (n_mels, time)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-9)  # Normalize to [0, 1]
        mel = mel.unsqueeze(0)  # Add channel dim
        mel = mel[:, :, :100] if mel.shape[2] >= 100 else F.pad(mel, (0, 100 - mel.shape[2]))
        spike_input = ttfs_encode(mel)
        tensors.append(spike_input)
        targets.append(label_to_index[label])
    return torch.stack(tensors), torch.tensor(targets)

# Loaders
train_loader = DataLoader(SubsetSC("training"), batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(SubsetSC("validation"), batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(SubsetSC("testing"), batch_size=64, shuffle=False, collate_fn=collate_fn)

# Network Definition
beta = 0.95
spike_grad = surrogate.fast_sigmoid()

class TTFSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 5 * 12, 512)
        self.drop1 = nn.Dropout(0.5)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(512, 35)

    def forward(self, x):
        mem1, mem2, mem3, mem4 = 0, 0, 0, 0
        spk_out = []
        for step in range(num_steps):
            x_t = x[:, :, :, :, step]  # Shape: [B, C, H, W]
            cur1 = self.conv1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            out1 = self.pool1(spk1)

            cur2 = self.conv2(out1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out2 = self.pool2(spk2)

            cur3 = self.conv3(out2)
            spk3, mem3 = self.lif3(cur3, mem3)
            out3 = self.pool3(spk3)

            flat = self.flatten(out3)
            cur4 = self.fc1(flat)
            drop = self.drop1(cur4)
            spk4, mem4 = self.lif4(drop, mem4)

            out = self.fc2(spk4)
            spk_out.append(out)

        return torch.stack(spk_out, dim=0)

# Model, Loss, Optimizer
model = TTFSNet().to(device)
loss_fn = SF.ce_temporal_loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training Loop
def train(epoch):
    model.train()
    total_loss, total_correct = 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss_val = loss_fn(out, targets)
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()
        pred = out.sum(0).argmax(1)
        total_correct += (pred == targets).sum().item()

    acc = total_correct / len(train_loader.dataset)
    print(f"Epoch {epoch}: Train Loss = {total_loss:.4f}, Accuracy = {acc*100:.2f}%")

# Validation Loop
def validate():
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            out = model(data)
            pred = out.sum(0).argmax(1)
            total_correct += (pred == targets).sum().item()
    acc = total_correct / len(val_loader.dataset)
    print(f"Validation Accuracy: {acc*100:.2f}%")

# Run Training
for epoch in range(1, 51):
    train(epoch)
    if epoch % 5 == 0:
        validate()
