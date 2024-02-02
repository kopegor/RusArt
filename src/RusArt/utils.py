import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import os

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import time
from tqdm import tqdm

import streamlit as st


def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value


def train_model(model, dataloaders, criterion, optimizer,
                phases, num_epochs=3, device="cpu", gamma=1):

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
    start_time = time.time()

    acc_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = len(dataloaders[phase])
            for inputs, labels in tqdm(dataloaders[phase], total=n_batches):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
              scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()
            epoch_acc /= len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

        torch.save(model.state_dict(), MODEL_WEIGHTS)

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))

    return model, acc_history


@st.cache_data
def init_model(device, num_classes):
    model = torchvision.models.swin_t(pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(device)
    return model


@st.cache_data
class ArtDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):

        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["label_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target


def get_category(image, MODEL_WEIGHTS):
    model = init_model("cpu", num_classes=35)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
    model.eval()

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = trans(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        pred = pred.numpy()

    labels = pd.read_csv(
        "C:\\Users\Egor\Projects\Russian_art_hack\RusArt\src\RusArt\label_to_id.csv",
        sep="\t")
    art_type = labels[labels["label_id"] == pred[0]].values[0, 0]

    return art_type


def main():
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dset = ArtDataset(TRAIN_DATASET, TRAIN_CSV, trans)
    labels = dset.targets
    indices = list(range(len(labels)))
    ind_train, ind_test, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=139, stratify=labels)

    trainset = torch.utils.data.Subset(dset, ind_train)
    testset = torch.utils.data.Subset(dset, ind_test)

    batch_size = 100
    num_workers = 2
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    loaders = {'train': trainloader, 'val': testloader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = init_model(device, num_classes=35)
    set_requires_grad(model, True)

    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    # Pretraine fine tuning
    pretrain_results = train_model(model, loaders, criterion, pretrain_optimizer,
                                   phases=['train', 'val'], num_epochs=9, device=device, gamma=1)

    torch.save(model.state_dict(), MODEL_WEIGHTS)

    # Train
    # запустить дообучение модели
    set_requires_grad(model, False)
    set_requires_grad(model.head, True)
    train_results = train_model(model, loaders, criterion, train_optimizer,
                                phases=['train', 'val'], num_epochs=9, device=device, gamma=0.85)

    torch.save(model.state_dict(), MODEL_WEIGHTS)



