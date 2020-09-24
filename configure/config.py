#-*-coding:utf-8-*-
import os
import sys
import time
##############preprocessor################
home = "/users/zhouyanling/"
image_path= home + "Datasets"
class_image_path = image_path + "/imagenette2-160-c/"
image_train_path = class_image_path + "train"
image_test_path = class_image_path + "test"

val_ration = .2
assert val_ration < 1

batch_size = 128

##################train###################
net_name = [["resnet34", "resnet50", "resnet101"],
            ["vgg11", "vgg13", "vgg16", "vgg19"]]

lr = 1e-2

max_epoch = 500

num_classes=10

project_path = home + "PycharmProjects/"

class_project_path = project_path + "ImageClassfication/"

outputs_path = class_project_path + "outputs/"

if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

class_outputs_path = outputs_path + net_name[1][3]

if not os.path.exists(class_outputs_path):
    os.mkdir(class_outputs_path)

json_path = outputs_path +"class_indices.json"

today = time.strftime("%Y-%m-%d",time.localtime(time.time()))

save_mpdel_path = class_outputs_path + "/" + today + "-" #+ sys.argv[1]

log_path = save_mpdel_path +"/info.log"

if not os.path.exists(save_mpdel_path):
    os.mkdir(save_mpdel_path)

if not os.path.exists(log_path):
    open(log_path,'w')

##################predict###################

# predict image path
#this is temp directory
predict_path = class_outputs_path + "1.jpg"
