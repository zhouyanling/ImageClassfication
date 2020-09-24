#-*-coding:utf-8-*-

import sys
import torch
import yaml

from models import resnet
#from configure import config as cfg
from data_processor import preprocess

#get yaml content
yamlPath = "/users/zhouyanling/PycharmProjects/ImageClassfication/configure/lenet.yaml"
yaml_file = open(yamlPath,'r',encoding='utf-8')
content = yaml_file.read()
cfg_content = yaml.load(content)

device = torch.device(cfg_content['device'] if torch.cuda.is_available() else "cpu")

###################################### manuall modify###############################
net = resnet.resnet34(cfg_content['train']['num_classes'])
###################################### manuall modify###############################

net.to(device)
model_weight_path = preprocess.save_mpdel_path + sys.argv[1]
net.load_state_dict(torch.load(model_weight_path))


acc = 0.0
best_acc = 0.0
with torch.no_grad():
    for test_data in preprocess.test_loader:
        test_images, test_labels = test_data
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = net(test_images)
        predict_y = torch.max(outputs,dim=1)[1]
        acc += (predict_y == test_labels.to(device)).sum().item()

    val_accurate = acc / preprocess.test_len

    print("test_accuracy: %.5f" %(val_accurate))