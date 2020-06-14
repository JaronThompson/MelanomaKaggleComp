
# coding: utf-8

# In[1]:


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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import xgboost as xgb
from efficientnet_pytorch import EfficientNet


# In[2]:


# Device configuration (GPU can be enabled in settings)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)


# In[3]:

all = pd.read_csv("../TrainTestDataFrames/train_concat.csv")
train_df = pd.read_csv("train_df.csv")
val_df = pd.read_csv("val_df.csv")
path = "../../data/train/train/"

print("Training on {} images, validating on {} images.".format(train_df.shape[0], val_df.shape[0]))


# In[4]:


# Convolutional neural network
class MyENet(nn.Module):
    def __init__(self, Net):
        super(MyENet, self).__init__()
        self.Net = Net
        self.output = nn.Sequential(
            nn.Linear(1000, 1),
            nn.Sigmoid())

    def embedding(self, x):
        out = self.Net(x)
        return out

    def forward(self, x):
        out = self.Net(x)
        out = self.output(out)
        return out

# First, load the EfficientNet with pre-trained parameters
ENet = EfficientNet.from_pretrained('efficientnet-b0').to(device)
model = MyENet(ENet).to(device)
model.load_state_dict(torch.load('../Models/ENetmodel.ckpt'), strict=False)


# In[5]:


meta_features = ['sex', 'age_approx', 'anatom_site_general_challenge']

encoder = {}
for feature in meta_features:
    # determine unique features
    categories = np.unique(np.array(all[feature].values, str))
    for i, category in enumerate(categories):
        if category != 'nan':
            encoder[category] = np.float(i)
encoder['nan'] = np.nan

transform = transforms.Compose(
                   [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor()])

class Dataset(torch.utils.data.Dataset):
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

        # load y
        label = self.df["target"].values[index]
        target = torch.tensor(label, dtype=torch.float32)

        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        #img = img.resize((256, 256))
        img_processed = transform(img)
        # 3. Return a data pair (e.g. image and label).
        return img_processed, meta_data, target

    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]


# In[6]:


# Use the data loader.
'''
batch_size = 16

train_dataset = Dataset(train_df, path)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size)

model = model.eval()
for i, (images, meta_data, labels) in enumerate(train_loader):
    images = images.to(device)

    # Forward pass
    embed = model.embedding(images)
    nn_pred = model.output(embed).detach().cpu().numpy()
    embedding = embed.detach().cpu().numpy()

    # determine NN features for the set of images
    batch_features = np.concatenate((embedding, meta_data.numpy(), nn_pred), axis=1)

    # append the dataset
    try:
        X = np.concatenate((X, batch_features), 0)
        y = np.append(y, labels.numpy())
    except:
        X = batch_features
        y = labels.numpy()

# In[7]:


# Save X and y in pandas dataframe
XGB_data = pd.DataFrame(data=X)
XGB_data['targets'] = y
XGB_data.to_csv("XGB_train.csv", index=False)
'''
XGB_data = pd.read_csv("XGB_train.csv")

X = np.array(XGB_data.values[:, :-1], np.float32)
y = np.array(XGB_data['targets'].values, np.float32)


# In[8]:


# weight positive examples more heavily
def make_weights(targets, nclasses=2):
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


# In[13]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=21)

# weight positive examples more heavily
w = make_weights(y_train)

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w)
dval = xgb.DMatrix(X_val, label=y_val)

# booster params
param = {'max_depth': 16, 'eta': .05, 'objective': 'binary:logistic'}
param['nthread'] = 8
param['eval_metric'] = 'auc'
#param['subsample'] = .75
#param['gpu_id'] = 0
#param['tree_method'] = 'gpu_hist'

# specify validation set
evallist = [(dval, 'eval')]

# Training
num_round = 5000
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=15)


# In[22]:
batch_size=16
valid_dataset = Dataset(val_df, path)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size)

model = model.eval()
for i, (images, meta_data, labels) in enumerate(valid_loader):
    images = images.to(device)

    # Forward pass
    embed = model.embedding(images)
    nn_pred = model.output(embed).detach().cpu().numpy()
    embedding = embed.detach().cpu().numpy()

    # determine NN features for the set of images
    batch_features = np.concatenate((embedding, meta_data.numpy(), nn_pred), axis=1)

    # append the dataset
    try:
        Xval = np.concatenate((Xval, batch_features), 0)
        yval = np.append(yval, labels.numpy())
    except:
        Xval = batch_features
        yval = labels.numpy()

XGB_data = pd.DataFrame(data=Xval)
XGB_data['targets'] = yval
XGB_data.to_csv("XGB_val.csv", index=False)

#XGB_data = pd.read_csv("XGB_val.csv")
Xval = np.array(XGB_data.values[:, :-1], np.float32)
yval = np.array(XGB_data['targets'].values, np.float32)


# In[19]:


dtest = xgb.DMatrix(Xval)
ypred = bst.predict(dtest)

fpr, tpr, _ = roc_curve(yval, ypred)
roc_auc = auc(fpr, tpr)

plt.style.use('seaborn-colorblind')
plt.rcParams.update({'font.size': 16,
                     'legend.framealpha':1,
                     'legend.edgecolor':'inherit'})
plt.figure(figsize=(9, 6))

lw = 2
plt.plot(fpr, tpr,
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("Figures/XGB_ROC_on_val.png")
plt.close()


# In[17]:


val_acc = accuracy_score(yval, np.round(ypred))
print("validation accuracy: {:.3f}".format(val_acc))


# In[ ]:

bst.save_model("../Models/xgbNN.model")
