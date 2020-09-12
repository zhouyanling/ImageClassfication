#-*-coding:utf-8-*-

import sys
import torch
from models import resnet
from configure import config as cfg
from data_processor import preprocess

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


net = resnet.resnet34(cfg.num_classes)
net.to(device)
model_weight_path = cfg.model_path + sys.argv[1]
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