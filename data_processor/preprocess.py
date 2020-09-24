#-*-coding:utf-8-*-
import os
import sys
import yaml
import json
import torch
import time
from torch.utils.data import Dataset
from torchvision import transforms,datasets

sys.path.append("..")

#from configure import config as cfg

#get yaml content
yamlPath = "/users/zhouyanling/PycharmProjects/ImageClassfication/configure/alexnet.yaml"
yaml_file = open(yamlPath,'r',encoding='utf-8')
content = yaml_file.read()
cfg_content = yaml.load(content)

#check directory
if not os.path.exists(cfg_content['train']['outputs_path']):
    os.mkdir(cfg_content['train']['outputs_path'])

class_outputs_path = cfg_content['train']['outputs_path'] + cfg_content['model']['name']

if not os.path.exists(class_outputs_path):
    os.mkdir(class_outputs_path)

today = time.strftime("%Y-%m-%d",time.localtime(time.time()))

save_mpdel_path = class_outputs_path + "/" + today + "-" + sys.argv[1]

predict_path = class_outputs_path + "1.jpg"

log_path = save_mpdel_path +"/info.log"

if not os.path.exists(save_mpdel_path):
    os.mkdir(save_mpdel_path)

if not os.path.exists(log_path):
    open(log_path,'w')

#define SubsetDataset to split training data into train and validate data
class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):

        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(cfg_content['dataset']['image_size']),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    "val": transforms.Compose([transforms.Resize(cfg_content['dataset']['val_size']),
                               transforms.CenterCrop(cfg_content['dataset']['image_size']),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
}


train_dataset = datasets.ImageFolder(cfg_content['dataset']['train_dir'])

# write class name to json file, for signal picture predict
image_list = train_dataset.class_to_idx
cla_dict = dict((val,key) for key,val in image_list.items())
json_str = json.dumps(cla_dict,indent=4)
json_path = cfg_content['dataset']['json_path']
with open(json_path,'w') as json_file:
    json_file.write(json_str)


total_size = len(train_dataset)

val_size = int(len(train_dataset) * cfg_content['dataset']['val_ration'])
train_size = total_size -val_size
lengths = [train_size,val_size]
train_subset, val_subset = torch.utils.data.dataset.random_split(train_dataset,lengths)

#train  and validate dataset
train_subset = SubsetDataset(train_subset, data_transform['train'])
val_subset = SubsetDataset(val_subset, data_transform['val'])
train_len = len(train_subset)
val_len = len(val_subset)

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=cfg_content['dataset']['batch_size'],
                                           shuffle=True, num_workers=cfg_content['dataset']['num_workers'])
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=cfg_content['dataset']['batch_size'],
                                         shuffle=True, num_workers=cfg_content['dataset']['num_workers'])

#test dataset
test_dataset = datasets.ImageFolder(root=cfg_content['dataset']['test_dir'],transform=data_transform['val'])
test_len = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg_content['dataset']['batch_size'],
                                          shuffle=True,num_workers=cfg_content['dataset']['num_workers'])
