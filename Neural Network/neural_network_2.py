from datetime import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time

def addNoise(x, mean, std):
    return x + torch.distributions.Normal(mean, std).sample(sample_shape=torch.Size(x.shape))


class AdditiveGausNoise(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            return addNoise(x, self.mean, self.std)
        else:
            return x


class AddGaussianNoise_1(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NeuralNetwork(nn.Module):
    def __init__(self, denoising, mean, std):
        super(NeuralNetwork, self).__init__()
        self.list_of_layers = nn.ModuleList()
        if denoising:
            self.encoder = nn.Sequential(
                AdditiveGausNoise(mean, std),
                nn.Conv2d(1, 2, (3, 3), padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Conv2d(2, 2, (3, 3), padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Conv2d(2, 2, (3, 3), padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Conv2d(2, 2, (3, 3), padding=1)
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(2, 2, (3, 3), padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Conv2d(2, 2, (3, 3), padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Conv2d(2, 2, (3, 3), padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Conv2d(2, 1, (3, 3), padding=1)
            )
            self.list_of_layers.append(self.encoder)
            self.list_of_layers.append(self.decoder)
        else:
            self.list_of_layers.append(nn.Flatten())
            self.list_of_layers.append(nn.Sequential(
                nn.Linear(784, 500),
                nn.SELU(),
                nn.BatchNorm1d(500),
                nn.Linear(500, 256),
                nn.Sigmoid(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 10),
                nn.SELU(),
                nn.BatchNorm1d(10),
            ))
            self.list_of_layers.append(nn.Softmax(dim=0))

    def forward(self, x):
        for layer in self.list_of_layers:
            x = layer(x)
        return x


class AutoEncodeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)
        return x, x


def train(denoising, tag, num, disable_tqdm=False):
    if denoising:
        transformation = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        train_data = AutoEncodeDataset(
            torchvision.datasets.MNIST("./data", train=True, transform=transformation, download=False))
        test_data_xy = torchvision.datasets.MNIST("./data", train=False, transform=transformation,
                                                  download=False)
        test_data_xx = AutoEncodeDataset(test_data_xy)
        trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
        testdata = DataLoader(test_data_xx, batch_size=128)
        score_funcs = {}
        to_track = ["epoch", "total time", "train loss"]
        for eval_score in score_funcs:
            to_track.append("train " + eval_score)
            if testdata is not None:
                to_track.append("val " + eval_score)
        results = {}
        for item in to_track:
            results[item] = []
        total_train_time = 0
        model = None
        if tag:
            model = NeuralNetwork(True, denoising[0], denoising[1])
            criterion = nn.MSELoss()
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
            model = model.train()
            epochs = 40
            start = time.time()
            for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
                model = model.train()
                running_loss = 0.0
                y_true = []
                y_pred = []
                for inputs, labels in tqdm(trainloader, desc="Train Batch", leave=False, disable=disable_tqdm):
                    batch_size = labels.shape[0]
                    optimizer.zero_grad()
                    y_hat = model(inputs)
                    loss = criterion(y_hat, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    running_loss = 0
                    running_loss += loss.item() * batch_size
                    if len(score_funcs) > 0:
                        labels = labels.detach().cpu().numpy()
                        y_hat = y_hat.detach().cpu().numpy()
                        for i in range(batch_size):
                            y_true.append(labels[i])
                            y_pred.append(y_hat[i, :])
                end = time.time()
                total_train_time += (end - start)
                results["epoch"].append(epoch)
                results["total time"].append(total_train_time)
                results["train loss"].append(running_loss)
                y_pred = np.asarray(y_pred)
                if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                for name, score_func in score_funcs.items():
                    results["train " + name].append(score_func(y_true, y_pred))
            torch.save(model, 'mod' + str(num) + '.pth')
        return model, testdata
    else:
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transformation)
        train_set, valid_set = torch.utils.data.random_split(train_set, lengths=[50000, 10000], generator=torch.Generator().manual_seed(42))
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transformation)
        train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_set, batch_size=50, shuffle=True, num_workers=2)
        testloader = DataLoader(test_set, batch_size=30, shuffle=False, num_workers=2)
        model = None
        if tag:
            model = NeuralNetwork(False, 0, 0)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
            num_epochs = 500
            best_vloss = 1_000_000
            for epoch in range(num_epochs):
                model.train(True)
                running_loss = 0.
                last_loss = 0.
                for idx, data in enumerate(train_loader, 0):
                    Xb, yb = data
                    optimizer.zero_grad()
                    pred = model(Xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if idx % 30 == 29:
                        last_loss = running_loss / 30
                        print(f'[{epoch + 1}, {idx + 1:5d}] loss: {last_loss:.3f}')
                        running_loss = 0.0
                model.train(False)
                running_vloss = 0.0
                for i, vdata in enumerate(valid_loader):
                    vinputs, vlabels = vdata
                    voutputs = model(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    running_vloss += vloss
                avg_vloss = running_vloss / (i + 1)
                print('LOSS train {} valid {}'.format(last_loss, avg_vloss))
                if last_loss < best_vloss:
                    best_vloss = last_loss
                    torch.save(model, 'model.pth')
        return model, testloader


def predict(model, test_loader):
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    model.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5d} is {accuracy:.1f} %')


def showEncodeDecode(original_copy, noisy, out):
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(original_copy, cmap='gray')
    axarr[0].set_title("Original Image")
    axarr[1].imshow(noisy, cmap='gray')
    axarr[1].set_title("Noisy Image")
    axarr[2].imshow(out, cmap='gray')
    axarr[2].set_title("Cleaned Image")
    f.show()


def denoise(model, testloader, mean, std):
    model = model.eval()
    tr_1 = transforms.Compose([
        AddGaussianNoise_1(mean, std)
    ])
    for idx, data in enumerate(testloader):
        original, labels = data
        original_copy = original
        noisy = tr_1(original)
        out = model(noisy)
        out = out * 0.3081 + 0.1307
        original_copy = original_copy[0, 0].detach().cpu().numpy()
        noisy = noisy[0, 0].detach().cpu().numpy()
        out = out[0, 0].detach().cpu().numpy()
        showEncodeDecode(original_copy, noisy, out)





if __name__ == '__main__':
    model_schema = train(None, False, 0)
    predict(torch.load('model.pth'), model_schema[1])
    for noise in [(0, 2, 2), (0, 0.5, 0), (0, 1, 1)]:
        model_schema = train((noise[0], noise[1]), False, noise[2])
        denoise(torch.load('mod' + str(noise[2]) + '.pth'), model_schema[1], noise[1], noise[2])