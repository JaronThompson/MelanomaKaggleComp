import numpy as np
import pandas as pd
#import pydicom
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import xgboost as xgb
from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

#%%

# Device configuration (GPU can be enabled in settings)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

#%%

train_df = pd.read_csv("../TrainTestDataFrames/marking.csv")
test_df = pd.read_csv("../TrainTestDataFrames/test.csv")

train_path = "../../data-512/512x512-dataset-melanoma/512x512-dataset-melanoma/"
test_path  = "../../data-512/512x512-test/512x512-test/"

#%% First, load the EfficientNet with pre-trained parameters

ENet = EfficientNet.from_pretrained('efficientnet-b0').to(device)

# Convolutional neural network
class MyENet(nn.Module):
    def __init__(self, ENet):
        super(MyENet, self).__init__()
        # modify output layer of the pre-trained ENet
        self.ENet = ENet
        num_ftrs = self.ENet._fc.in_features
        self.ENet._fc = nn.Linear(in_features=num_ftrs, out_features=512, bias=True)
        # map Enet output to melanoma decision
        self.output = nn.Sequential(nn.BatchNorm1d(512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(512, 1),
                                    nn.Sigmoid())

    def embedding(self, x):
        out = self.ENet(x)
        return out

    def forward(self, x):
        out = self.ENet(x)
        out = self.output(out)
        return out

model = MyENet(ENet).to(device)
# Load best model
model.load_state_dict(torch.load('../Models/ENETmodel_all.ckpt'))
model = model.eval()

#%%

meta_features = ['sex', 'age_approx', 'anatom_site_general_challenge']

encoder = {}
for feature in meta_features:
    # determine unique features
    categories = np.unique(np.array(train_df[feature].values, str))
    for i, category in enumerate(categories):
        if category != 'nan':
            encoder[category] = np.float(i)
encoder['nan'] = np.nan

# train transform adds noise
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# no flip or rotation for test/validation data
test_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, path_to_files):
        # 1. Initialize file paths or a list of file names.
        self.path = path_to_files
        self.df = df

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        # load X
        img_name = self.df['image_id'].values[index]
        img_path = self.path + img_name + ".jpg"
        img = plt.imread(img_path)

        # determine meta data
        meta = self.df[meta_features].values[index]
        meta_data = np.array([encoder[str(m)] for m in meta])

        # load y
        label = self.df["target"].values[index]
        target = torch.tensor(label, dtype=torch.float32)

        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        img_processed = train_transform(img)
        # 3. Return a data pair (e.g. image and label).
        return img_processed, meta_data, target

    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, path_to_files):
        # 1. Initialize file paths or a list of file names.
        self.path = path_to_files
        self.df = df

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        # load X
        img_name = self.df['image_name'].values[index]
        img_path = self.path + img_name + ".jpg"
        img = plt.imread(img_path)

        # determine meta data
        meta = self.df[meta_features].values[index]
        meta_data = np.array([encoder[str(m)] for m in meta])

        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        img_processed = test_transform(img)
        # 3. Return a data pair (e.g. image and label).
        return img_processed, meta_data, img_name

    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]

#%% set up training data
batch_size = 12
'''
train_dataset = TrainDataset(train_df, train_path)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

for i, (images, meta_data, labels) in enumerate(tqdm(train_loader)):
    images = images.to(device)

    # Forward pass
    embed = model.embedding(images)
    nn_pred = model.output(embed).detach().cpu().numpy()
    embedding = embed.detach().cpu().numpy()

    # determine NN features for the set of images
    batch_features = np.concatenate((embedding, meta_data, nn_pred), axis=1)

    # append the dataset
    try:
        X = np.concatenate((X, batch_features), 0)
        y = np.append(y, labels.numpy())
    except:
        X = batch_features
        y = labels.numpy()

# Save X and y in pandas dataframe just in case
XGB_data = pd.DataFrame(data=X)
XGB_data['targets'] = y
XGB_data.to_csv("XGB_ENET_train_all.csv", index=False)
'''
XGB_data = pd.read_csv("XGB_ENET_train_all.csv")
X = np.array(XGB_data.values[:, :-1], np.float32)
y = np.array(XGB_data['targets'].values, np.float32)

# Get stats for normalizing training and testing data
mean_X = np.nanmean(X, 0)
std_X = np.nanstd(X, 0)

# standardize training data set
X_train_std = (X - mean_X) / std_X

#%% set up test data

test_dataset = TestDataset(test_df, test_path)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = model.eval()
for i, (images, meta_data, batch_image_names) in enumerate(tqdm(test_loader)):

    images = images.to(device)

    # Forward pass
    embed = model.embedding(images)
    nn_pred = model.output(embed).detach().cpu().numpy()
    embedding = embed.detach().cpu().numpy()

    # determine NN features for the set of images
    batch_features = np.concatenate((embedding, meta_data.numpy(), nn_pred), axis=1)

    # append the dataset
    try:
        X_test = np.concatenate((X_test, batch_features), 0)
        image_names = np.append(image_names, batch_image_names)
    except:
        X_test = batch_features
        image_names = np.array(batch_image_names)

X_test_std = (X_test - mean_X) / std_X

#%% Define functions for fitting xgb model

# weight positive examples more heavily
def make_weights(targets):
    nclasses = len(np.unique(targets))
    count = [0] * nclasses
    for label in targets:
        count[np.int(label)] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(targets)
    for idx, label in enumerate(targets):
        weight[idx] = weight_per_class[np.int(label)]

    return np.array(weight)

# define function to fit and return xgboost model
def fit_xgboost(X_train, y_train, X_val, y_val):

    # weight positive examples more heavily
    w = make_weights(y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w)
    dval = xgb.DMatrix(X_val, label=y_val)

    # booster params
    param = {'n_estimators':5000,
            'max_depth':16,
            'learning_rate':0.02,
            'subsample':0.8,
            'eval_metric':'auc',
            'objective': 'binary:logistic',
            'nthread': 8}

    # specify validation set
    evallist = [(dval, 'eval')]

    # Training
    num_round = 5000
    bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=50)

    return bst

#%% Train kfolds and test

n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True)
skf.get_n_splits(X_train_std, y)

# define "out of fold" set of predictions, represents validation performance
oof = np.zeros(len(X_train_std))
predictions = np.zeros(len(X_test_std))

for i, (train_index, val_index) in enumerate(skf.split(X_train_std, y)):

    # get data partitions for Xtrain and Xval
    X_train, X_val = X_train_std[train_index], X_train_std[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # train xgboost
    bst = fit_xgboost(X_train, y_train, X_val, y_val)

    # save out of fold predictions
    oof[val_index] += bst.predict(xgb.DMatrix(X_val))

    # save current model predictions on the true validation set
    predictions += bst.predict(xgb.DMatrix(X_test_std)) / skf.n_splits

#%% Make submission file

submission = pd.DataFrame()
submission["image_name"] = image_names
submission["target"] = predictions
submission.to_csv("JT_submission_meta.csv", index=False)

#%% Estimate score based on OOF predictions

tn, fp, fn, tp = confusion_matrix(y, np.round(oof)).ravel()

accuracy = (tp + tn) / len(y)
# precision is the fraction of correctly identified positive samples
# precision asks:"Of all the samples identified as positives, how many were correct?"
precision = tp / (tp + fp)
# recall is the ability of the model to identify positive samples
# recall asks:"Of all the positive samples in the dataset, how many were identified by the model?"
recall = tp / (tp + fn)

print("Out Of Fold stats:")

print("Model accuracy: {:.2f}".format(accuracy))
print("Model precision: {:.2f}".format(precision))
print("Model recall: {:.2f}".format(recall))

print("\nConfusion Matrix: ")
print(confusion_matrix(y,  np.round(oof)))
