import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class PrepareData(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.mu_y = torch.mean(y)
        self.std_y = torch.std(y)
        self.mu_x = torch.mean(X)
        self.std_x = torch.std(X)
        self.X = (self.X - self.mu_x) / self.std_x
        self.y = (self.y - self.mu_y) / self.std_y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        return tensor + torch.distributions.Normal(self.mean, self.std).sample(sample_shape=torch.Size(tensor.shape))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, noise):
        super(NeuralNetwork, self).__init__()
        self.list_of_layers = nn.ModuleList()
        if noise:
            self.list_of_layers.append(AdditiveGausNoise(noise[0], noise[1]))
            self.list_of_layers.append(nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.GELU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.Sigmoid(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 1),
                nn.GELU(),
                nn.BatchNorm1d(1),
            ))
        else:
            self.list_of_layers.append(nn.Sequential(
                nn.Linear(input_size, 758),
                nn.SELU(),
                nn.BatchNorm1d(758),
                nn.Linear(758, 500),
                nn.SELU(),
                nn.BatchNorm1d(500),
                nn.Linear(500, 256),
                nn.Sigmoid(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 1),
                nn.BatchNorm1d(1),
            ))

    def forward(self, x):
        for layer in self.list_of_layers:
            x = layer(x)
        return x


def train(x, y, model_num, noise, tag):
    train, test = train_test_split(list(range(x.shape[0])), test_size=.3)
    ds = PrepareData(x, y)
    train_set = DataLoader(ds, batch_size=64,
                           sampler=SubsetRandomSampler(train))
    test_set = DataLoader(ds, batch_size=64,
                          sampler=SubsetRandomSampler(test))
    model = None
    criterion = torch.nn.MSELoss()
    if tag:
        model = NeuralNetwork(x.shape[1], noise)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.05, momentum=0.5)
        num_epochs = 500
        ##
        all_losses = []
        for e in range(num_epochs):
            batch_losses = []

            for idx, (Xb, yb) in enumerate(train_set):
                _X = Xb.float()
                _y = yb.float()
                pred = model(_X)
                loss = criterion(pred, _y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                all_losses.append(loss.item())

            mbl = torch.mean(torch.sqrt(torch.tensor(batch_losses)))

            if e % 5 == 0:
                print("Epoch [{}/{}], Batch loss: {}".format(e, num_epochs, mbl))
        torch.save(model, 'model_' + str(model_num) + '.pth')
    return model, criterion, test_set, ds


def predict(model, criterion, test_set, ds, model_num, noise_or_not):
    print(model.training)
    model.eval()
    print(model.training)
    test_x = torch.tensor([])
    test_pred_y = torch.tensor([])
    real_y = torch.tensor([])
    test_batch_losses = []
    noisal = torch.tensor([])
    for _X, _y in test_set:
        _X = _X.float()
        _y = _y.float()
        if noise_or_not:
            noisy = transforms.Compose([AddGaussianNoise_1(noise_or_not[0], noise_or_not[1])])(_X)
            test_pred = model(noisy)
        else:
            test_pred = model(_X)
        test_loss = criterion(test_pred, _y)
        test_batch_losses.append(test_loss.item())
        print("Batch loss: {}".format(test_loss.item()))
        if not model_num == 6:
            if _X.shape[0] == 40 and _y.shape[0] == 40:
                test_x = torch.hstack((torch.reshape(_X, (40,)) * ds.std_x + ds.mu_x, test_x))
                test_pred_y = torch.hstack((torch.reshape(test_pred, (40,)) * ds.std_y + ds.mu_y, test_pred_y))
                real_y = torch.hstack((torch.reshape(_y, (40,)) * ds.std_y + ds.mu_y, real_y))
                if noise_or_not:
                    noisal = torch.hstack((torch.reshape(noisy, (40,)) * ds.std_y + ds.mu_y, noisal))
            else:
                test_x = torch.hstack((torch.reshape(_X, (64,)) * ds.std_x + ds.mu_x, test_x))
                test_pred_y = torch.hstack((torch.reshape(test_pred, (64,)) * ds.std_y + ds.mu_y, test_pred_y))
                real_y = torch.hstack((torch.reshape(_y, (64,)) * ds.std_y + ds.mu_y, real_y))
                if noise_or_not:
                    noisal = torch.hstack((torch.reshape(noisy, (64,)) * ds.std_y + ds.mu_y, noisal))
    if not model_num == 6:
        plt.scatter(test_x.detach(), test_pred_y.detach())
        plt.scatter(test_x.detach(), real_y.detach())
        if noise_or_not:
            plt.scatter(test_x.detach(), noisal.detach())
            plt.legend(["predicted", "original", "noisy"])
        else:
            plt.legend(["predicted", "original"])
        plt.show()




x = torch.reshape(torch.linspace(-4 * torch.pi, 4 * torch.pi, 30000), (30000, 1))
y = torch.pow(x, 2)
tup_1 = train(x, y, 2, None, False)
predict(torch.load("model_1.pth"), tup_1[1], tup_1[2], tup_1[3], 1, None)


x = torch.reshape(torch.linspace(-4 * torch.pi, 4 * torch.pi, 30000), (30000, 1))
y = torch.exp(x) / (torch.sin(x) + 4)
tup_2 = train(x, y, 2, None, False)
predict(torch.load("model_2.pth"), tup_2[1], tup_2[2], tup_2[3], 2, None)


x = torch.reshape(torch.linspace(-4 * torch.pi, 4 * torch.pi, 30000), (30000, 1))
y = torch.exp((-4) * x) * (torch.sin(4 * x))
tup_3 = train(x, y, 3, None, False)
predict(torch.load("model_3.pth"), tup_3[1], tup_3[2], tup_3[3], 3, None)


x = torch.reshape(torch.linspace(-4 * torch.pi, 4 * torch.pi, 30000), (30000, 1))
y = torch.empty(30000, 1)
for i in range(len(x)):
    if x[i, 0] < 0:
        y[i, 0] = (-1) * x[i, 0] + 2
    elif 0 < x[i, 0] < 6:
        y[i, 0] = x[i, 0]
    else:
        y[i, 0] = 2 * x[i, 0]
tup_4 = train(x, y, 4, None, False)
predict(torch.load("model_4.pth"), tup_4[1], tup_4[2], tup_4[3], 4, None)

x_1 = torch.reshape(torch.linspace(-4 * torch.pi, 4 * torch.pi, 30000), (30000, 1))
x_2 = torch.reshape(torch.linspace(-4 * torch.pi, 4 * torch.pi, 30000), (30000, 1))
X = torch.hstack((x_1, x_2))
y = torch.pow(x_1, 2) + x_2
tup_5 = train(X, y, 6, None, False)
predict(torch.load("model_6.pth"), tup_5[1], tup_5[2], tup_5[3], 6, None)



for noise in [(10, 20, 8), (10, 10, 9), (0, 0.5, 7)]:
    x = torch.reshape(torch.linspace(-4 * torch.pi, 4 * torch.pi, 30000), (30000, 1))
    tup_4 = train(x, x, noise[2], [noise[0], noise[1]], False)
    predict(torch.load('model_' + str(noise[2]) + '.pth'), tup_4[1], tup_4[2], tup_4[3], 5, [noise[0], noise[1]])
    exit(0)







