
# coding: utf-8

# In[5]:


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


# In[3]:


# Device configuration (GPU can be enabled in settings)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)


# In[6]:


alldata = pd.read_csv("../TrainTestDataFrames/train_concat.csv")
test_df = pd.read_csv("../../data/test.csv")
path = "../../data/test/test/"


# In[7]:


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


# In[8]:


test_df.head()


# In[9]:


meta_features = ['sex', 'age_approx', 'anatom_site_general_challenge'] 

encoder = {}
for feature in meta_features: 
    # determine unique features  
    categories = np.unique(np.array(alldata[feature].values, str))
    for i, category in enumerate(categories): 
        if category != 'nan':
            encoder[category] = np.float(i)
encoder['nan'] = np.nan

transform = transforms.Compose([transforms.ToTensor()])

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
        #label = self.df["target"].values[index]
        #target = torch.tensor(label, dtype=torch.float32)
        
        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        #img = img.resize((256, 256))
        img_processed = transform(img)
        # 3. Return a data pair (e.g. image and label).
        return img_processed, meta_data, img_name
        
    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]


# In[11]:


test_dataset = Dataset(test_df, path)
                                            
batch_size = 16
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size,
                                          shuffle=False)   

model = model.eval()
for i, (images, meta_data, batch_image_names) in enumerate(test_loader):
    
    
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
        image_names = np.append(image_names, batch_image_names)
    except:
        X = batch_features 
        image_names = np.array(batch_image_names)


# In[8]:


bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('../Models/xgbNN.model')  # load data


# In[9]:


Dtest = xgb.DMatrix(X)
predictions = bst.predict(Dtest)


# In[10]:


submission = pd.DataFrame()

submission["image_name"] = image_names 
submission["target"] = predictions


# In[11]:


submission.to_csv("JT_submission_5.csv", index=False)

