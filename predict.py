#-*-coding:utf-8-*-
import sys

import torch
import torchvision.transforms as tranforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from models import resnet
from configure import config as cfg
from data_processor import preprocess

#load images
img = Image.open(cfg.predict_path)
plt.imshow(img)

#[N,C,H,W]
img = preprocess.data_transform['val'](img)
#img = data_transforms(img)
#expand batch dimension
img = torch.unsqueeze(img,dim=0)
#read class_indict
class_dict = 0
try:
    json_file = open(cfg.json_path,'r')
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

#create model
model = resnet.resnet34(cfg.num_classes)
#load model_weight
model_weight_path = cfg.save_mpdel_path + sys.argv[2]

model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim=0)
    predict_cla = torch.argmax(predict).numpy()

print(class_dict[str(predict_cla)],predict[predict_cla].numpy())
plt.show()