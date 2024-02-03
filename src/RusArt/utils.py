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


def set_requires_grad(model: torchvision.models, value=False):
    """
    Function to freeze or unfreeze model parameters

    :param model:
    :param value:
    :return:
    """
    # set requires_grad to given mode for every parameter in given model
    for param in model.parameters():
        param.requires_grad = value


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    phases,
    num_epochs=3,
    device="cpu",
    gamma=1.0,
):
    """
    Function to train neural network model

    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param phases:
    :param num_epochs:
    :param device:
    :param gamma:
    :return:
    """
    # initialize scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=gamma, verbose=True
    )
    # the beginning of training time calculation
    start_time = time.time()

    acc_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == "train":
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
                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # learning rate decay
            if phase == "train":
                scheduler.step()

            # calculation epoch's metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()
            epoch_acc /= len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            # collect metrics history
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

        # save model weights
        torch.save(model.state_dict(), MODEL_WEIGHTS)

        print()

    # calculate training time
    time_elapsed = time.time() - start_time
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model, acc_history


@st.cache_data
def init_model(device, num_classes):
    """
    Function to initialize neural network model
    :param device:
    :param num_classes:
    :return:
    """
    # load pretrained model
    model = torchvision.models.swin_t(pretrained=True)

    # modify last layer for our specific problem
    model.head = nn.Linear(model.head.in_features, num_classes)

    # move model to required device
    model = model.to(device)
    return model


@st.cache_data
class ArtDataset(Dataset):
    """
    Class for data loading and processing
    """

    def __init__(self, root_dir, csv_path=None, transform=None):
        # image transforms required for model input
        self.transform = transform

        # get file paths to load images
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

        # init targets
        self.targets = None

        # get images labels if they exists
        if csv_path:
            # load labels
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["label_id"].tolist()

            # update file paths
            self.files = [
                os.path.join(root_dir, fname) for fname in df["image_name"].tolist()
            ]

    def __len__(self):
        # get len of available dataset
        return len(self.files)

    def __getitem__(self, idx):
        # load required image
        image = Image.open(self.files[idx]).convert("RGB")

        # get target label for image
        target = self.targets[idx] if self.targets else -1

        # transform image to required format
        if self.transform:
            image = self.transform(image)

        return image, target


@st.cache_data
def get_category(image, MODEL_WEIGHTS):
    """
    Function to make prediction and get art category of image
    :param image:
    :param MODEL_WEIGHTS:
    :return:
    """
    # initialize trained model
    model = init_model("cpu", num_classes=35)

    # load model weights
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device("cpu")))

    # move model to eval mode
    model.eval()

    # init image transforms for model input
    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # transform image
    image = trans(image)

    # add one dimension, it's required for model input
    image = image.unsqueeze(0)

    # get prediction
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        pred = pred.numpy()

    # load labels of art categories
    labels = pd.read_csv(
        "C:\\Users\Egor\Projects\Russian_art_hack\RusArt\src\RusArt\label_to_id.csv",
        sep="\t",
    )
    # get art category from predicted label
    art_type = labels[labels["label_id"] == pred[0]].values[0, 0]

    return art_type


def main():
    """
    Function with whole pipeline of model training
    :return:
    """
    # define paths
    MODEL_WEIGHTS = "./baseline.pt"
    TRAIN_DATASET = "./data/train/"
    TRAIN_CSV = "./data/private_info/train.csv"

    # init image transforms
    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # create dataset
    dset = ArtDataset(TRAIN_DATASET, TRAIN_CSV, trans)

    # get labels
    labels = dset.targets
    indices = list(range(len(labels)))

    # get indices of train and test parts
    ind_train, ind_test, _, _ = train_test_split(
        indices, labels, test_size=0.2, random_state=139, stratify=labels
    )
    # split dataset to train and test parts
    trainset = torch.utils.data.Subset(dset, ind_train)
    testset = torch.utils.data.Subset(dset, ind_test)

    # init batch size
    batch_size = 100

    # init number of workers for parallelization
    num_workers = 2

    # create train dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    # create test dataloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    loaders = {"train": trainloader, "val": testloader}
    # get available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize model for training
    model = init_model(device, num_classes=35)

    # unfreeze all layers for pretrain model
    set_requires_grad(model, True)

    # init optimizer with small learning rate for pretrain
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # init optimizer with greater learning rate for fine-tuning
    train_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # define loss
    criterion = nn.CrossEntropyLoss()

    # Pretrain model for our specific problem
    pretrain_results = train_model(
        model,
        loaders,
        criterion,
        pretrain_optimizer,
        phases=["train", "val"],
        num_epochs=9,
        device=device,
        gamma=1,
    )
    # save model weights
    torch.save(model.state_dict(), MODEL_WEIGHTS)

    # Train model
    # freeze all layers
    set_requires_grad(model, False)

    # unfreeze model head for fine-tuning
    set_requires_grad(model.head, True)

    # train model
    train_results = train_model(
        model,
        loaders,
        criterion,
        train_optimizer,
        phases=["train", "val"],
        num_epochs=9,
        device=device,
        gamma=0.85,
    )
    # save final model weights
    torch.save(model.state_dict(), MODEL_WEIGHTS)
