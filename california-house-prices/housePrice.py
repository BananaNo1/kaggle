import csv

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# train_data = pd.read_csv('train.csv')
# ##  train_data.shape  (47439, 41)
#
#
# train_data.drop(["Id", "Address", "Summary", "Year built", "Heating", "Cooling", "Parking"
#                     , "Elementary School", "Middle School", "High School", "Flooring", "Heating features"
#                     , "Cooling features", "Appliances included", "Laundry features", "Parking features"
#                     , "Listed On", "Last Sold On", "State"], axis=1, inplace=True)
# train_data = train_data.iloc[:, 1:]
# train_data['Bedrooms'] = pd.to_numeric(train_data['Bedrooms'], errors="coerce")
#
# numeric_features = train_data.dtypes[train_data.dtypes != 'object'].index
# train_data[numeric_features] = train_data[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
#
# train_data[numeric_features] = train_data[numeric_features].fillna(0)


class houseDataSet(Dataset):
    def __init__(self, mode, path):
        self.mode = mode
        self.path = path
        data = pd.read_csv(path)
        data.drop(["Id", "Address", "Summary", "Year built", "Heating", "Cooling", "Parking"
                      , "Elementary School", "Middle School", "High School", "Flooring", "Heating features"
                      , "Cooling features", "Appliances included", "Laundry features", "Parking features"
                      , "Listed On", "Last Sold On", "State"], axis=1, inplace=True)
        data['Bedrooms'] = pd.to_numeric(data['Bedrooms'], errors="coerce")
        target_data = []
        if mode == "train":
            target_data = data.iloc[:, 0]
        data = data.iloc[:, 1:]
        numeric_features = data.dtypes[data.dtypes != 'object'].index
        data[numeric_features] = data[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
        data = data[numeric_features].fillna(0)

        self.data = torch.FloatTensor(data.values)
        self.target_data = torch.FloatTensor(target_data)
        self.target_data = self.target_data.view(len(target_data), 1)

    def __getitem__(self, item):
        if self.mode == "train":
            return self.data[item], self.target_data[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


def pre_dataloader(path, mode, batch_size, n_jobs):
    dataset = houseDataSet(mode, path)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=n_jobs)
    return dataloader


def train(tr_set, config, model, device):
    n_epochs = config['n_epochs']

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_record = []
    min_mse = 1000.

    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            print(pred)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record.append(mse_loss.detach().cpu().numpy())
        epoch += 1

        # print('Saving model (epoch = {:4d}, loss = {:.4f})'
        #       .format(epoch + 1, min_mse))
        # torch.save(model.state_dict(), config['save_path'])

        return loss_record


def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


config = {
    'n_epochs': 300,
    'batch_size': 4000,
    'save_path': 'models/model.pth'
}


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


tr_path = 'train.csv'
tt_path = 'test.csv'

device = get_device()
tr_set = pre_dataloader(tr_path, 'train', batch_size=config['batch_size'], n_jobs=0)
tt_set = pre_dataloader(tt_path, 'test', batch_size=config['batch_size'], n_jobs=0)

model = NeuralNet().to(device)
model_loss_record = train(tr_set, config, model, device)
# print(model_loss_record)

plt.title("loss")
plt.plot(list(range(len(model_loss_record))), model_loss_record)
plt.show()


# del model
# model = NeuralNet().to(device)
# ckpt = torch.load(config['save_path'], map_location='cpu')
# model.load_state_dict(ckpt)


def save_file(preds, file):
    print("Saving result to {}".format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'Sold Price'])
        for i, p in enumerate(preds):
            writer.writerow([47439 + i, p[0]])


preds = []

preds = test(tt_set, model, device)
save_file(preds, 'submission.csv')
