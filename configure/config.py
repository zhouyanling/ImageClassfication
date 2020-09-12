#-*-coding:utf-8-*-
import os
import sys
import time
##############preprocessor################
image_path= "/users/zhouyanling/Datasets/imagenette2-160-c/"
image_train_path = image_path + "train"
image_test_path = image_path + "test"

val_ration = .2
assert val_ration < 1

batch_size = 64

##################train###################

lr = 1e-3
max_epoch = 600
num_classes=10
model_path = "/users/zhouyanling/AIProject/Pytorch_RestNet/outputs/resnet34/"
json_path = model_path +"class_indices.json"

today = time.strftime("%Y-%m-%d",time.localtime(time.time()))

save_mpdel_path = model_path + today + "-" + sys.argv[1]

log_path = save_mpdel_path +"/info.log"

if not os.path.exists(save_mpdel_path):
    os.mkdir(save_mpdel_path)

if not os.path.exists(log_path):
    open(log_path,'w')

##################predict###################

# predict image path
#this is temp directory
predict_path = model_path + "1.jpg"