#-*-coding:utf-8-*-
import json
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms,datasets

sys.path.append("..")

from configure import config as cfg

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
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
}


train_dataset = datasets.ImageFolder(cfg.image_train_path)

# write class name to json file, for signal picture predict
image_list = train_dataset.class_to_idx
cla_dict = dict((val,key) for key,val in image_list.items())
json_str = json.dumps(cla_dict,indent=4)
json_path = cfg.json_path
with open(json_path,'w') as json_file:
    json_file.write(json_str)


total_size = len(train_dataset)

val_size = int(len(train_dataset) * cfg.val_ration)
train_size = total_size -val_size
lengths = [train_size,val_size]
train_subset, val_subset = torch.utils.data.dataset.random_split(train_dataset,lengths)

#train  and validate dataset
train_subset = SubsetDataset(train_subset, data_transform['train'])
val_subset = SubsetDataset(val_subset, data_transform['val'])
train_len = len(train_subset)
val_len = len(val_subset)

train_loader = torch.utils.data.DataLoader(train_subset,batch_size=cfg.batch_size,shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_subset,batch_size=cfg.batch_size,shuffle=True, num_workers=0)

#test dataset
test_dataset = datasets.ImageFolder(root=cfg.image_test_path,transform=data_transform['val'])
test_len = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=0)
