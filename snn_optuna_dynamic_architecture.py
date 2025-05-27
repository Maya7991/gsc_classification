import os
import sys
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from snntorch import spikegen, surrogate, functional as SF
import snntorch as snn
import optuna

batch_size = 64
num_epochs = 50
encoding_type = 'ttfs'
dataset_path = "../../../../datasets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

train_dataset = torchaudio.datasets.SPEECHCOMMANDS(dataset_path, download=False, subset="training")
val_dataset = torchaudio.datasets.SPEECHCOMMANDS(dataset_path, download=False, subset="validation")
test_dataset = torchaudio.datasets.SPEECHCOMMANDS(dataset_path, download=False, subset="testing")

all_labels = sorted(set(datapoint[2] for datapoint in train_dataset + val_dataset + test_dataset))
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
num_classes = len(label_encoder.classes_)

mel_transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 64})
target_length = 16000

def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        if waveform.size(1) < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.size(1)))
        else:
            waveform = waveform[:, :target_length]
        mfcc = mel_transform(waveform).squeeze(0)
        if encoding_type == 'ttfs':
            mfcc = (mfcc - mfcc.min()) / (mfcc.max() + 1e-6)
        else:
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-5)
        tensors.append(mfcc.unsqueeze(0))
        targets.append(label_encoder.transform([label])[0])
    return torch.stack(tensors), torch.tensor(targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def encode_input(x, num_steps):
    if encoding_type == 'rate':
        return spikegen.rate(x, num_steps=num_steps)
    elif encoding_type == 'ttfs':
        return spikegen.latency(x, num_steps=num_steps, normalize=True, linear=True)
    else:
        raise ValueError("Unknown encoding type")

def train(model, loader, optimizer, loss_fn, acc_fn, num_steps):
    model.train()
    total_loss, total_acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        spk_out = model(x, num_steps)
        loss = loss_fn(spk_out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc_fn(spk_out, y)
    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, loss_fn, acc_fn, num_steps):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            spk_out = model(x, num_steps)
            total_loss += loss_fn(spk_out, y).item()
            total_acc += acc_fn(spk_out, y)
    return total_loss / len(loader), total_acc / len(loader)

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    num_steps = trial.suggest_int("num_steps", 10, 50)
    beta = trial.suggest_uniform("beta", 0.7, 0.99)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 3)
    n_fc_layers = trial.suggest_int("n_fc_layers", 1, 3)

    conv_out_channels = [trial.suggest_categorical(f"conv{i}_out", [8, 16, 32, 64]) for i in range(n_conv_layers)]
    conv_kernels = [trial.suggest_int(f"conv{i}_kernel", 3, 7) for i in range(n_conv_layers)]
    fc_sizes = [trial.suggest_categorical(f"fc{i}_out", [64, 128, 256, 512, 1024]) for i in range(n_fc_layers)]

    loss_fn = SF.ce_temporal_loss() if encoding_type == "ttfs" else SF.ce_rate_loss()
    acc_fn = SF.accuracy_temporal if encoding_type == "ttfs" else SF.accuracy_rate

    class DynamicSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.lifs = nn.ModuleList()
            in_channels = 1
            for i in range(n_conv_layers):
                self.convs.append(nn.Conv2d(in_channels, conv_out_channels[i], kernel_size=conv_kernels[i]))
                self.lifs.append(snn.Leaky(beta=beta, threshold=0.9))
                in_channels = conv_out_channels[i]

            dummy = torch.zeros(1, 1, 40, 101)
            with torch.no_grad():
                for i in range(n_conv_layers):
                    dummy = F.max_pool2d(self.convs[i](dummy), 2)
                flat_dim = dummy.view(1, -1).size(1)

            self.fcs = nn.ModuleList()
            self.lif_fcs = nn.ModuleList()
            self.fcs.append(nn.Linear(flat_dim, fc_sizes[0]))
            self.lif_fcs.append(snn.Leaky(beta=beta, threshold=0.9))
            for i in range(1, n_fc_layers):
                self.fcs.append(nn.Linear(fc_sizes[i - 1], fc_sizes[i]))
                self.lif_fcs.append(snn.Leaky(beta=beta, threshold=0.9))

            self.out = nn.Linear(fc_sizes[-1], num_classes)
            self.out_lif = snn.Leaky(beta=beta, threshold=0.9)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, num_steps):
            for layer in self.lifs + self.lif_fcs + [self.out_lif]:
                layer.reset_mem()

            spk_out_rec = []
            cur_input = encode_input(x, num_steps)

            for step in range(num_steps):
                out = cur_input[step]
                for i in range(n_conv_layers):
                    out = F.max_pool2d(self.convs[i](out), 2)
                    out, _ = self.lifs[i](out)

                out = out.view(out.size(0), -1)
                for i in range(n_fc_layers):
                    out = self.dropout(self.fcs[i](out))
                    out, _ = self.lif_fcs[i](out)

                out = self.out(out)
                spk_out, _ = self.out_lif(out)
                spk_out_rec.append(spk_out)

            return torch.stack(spk_out_rec)

    model = DynamicSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, loss_fn, acc_fn, num_steps)
        _, val_acc = evaluate(model, val_loader, loss_fn, acc_fn, num_steps)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=7200)
    print("Best Trial:")
    print(f"  Accuracy: {100 * study.best_value:.2f}%")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")