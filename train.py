#-*-coding:utf-8-*-
import yaml
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
#from configure import config as cfg
#sys.path.append("..")

from models import resnet
from models import vgg
from models import lenet
from models import alexnet
from data_processor import preprocess

#get yaml content

cfg_content = preprocess.cfg_content

logging.basicConfig(level=logging.DEBUG,
                    filename=preprocess.log_path,
                    filemode='a',
                    format = '%(asctime)s - %(levelname)s:  %(message)s')

train_loader = preprocess.train_loader
val_loader = preprocess.val_loader
train_len = preprocess.train_len
val_len = preprocess.val_len
writer = SummaryWriter(preprocess.save_mpdel_path)
#data_root = os.getcwd()

device = torch.device(cfg_content['device'] if torch.cuda.is_available() else "cpu")

# load model######################### manuall modify###############################
#net = resnet.resnet34(cfg.num_classes)
#net = vgg.VGG(cfg_content['model']['name'], cfg_content['dataset']['n_classes'])
#net = lenet.LeNet(cfg_content['dataset']['n_classes'])
net = alexnet.AlexNet(cfg_content['dataset']['n_classes'])
###################################### manuall modify###############################

# load model
net.to(device)
loss_function = nn.CrossEntropyLoss()
#set optimizer

###################################### manuall modify###############################
#optimizer = optim.Adam(net.parameters(),lr=cfg.lr)
optimizer = optim.SGD(net.parameters(),
                       lr = float(cfg_content['train']['base_lr']),
                       momentum = cfg_content['train']['momentum'])


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode = cfg_content['scheduler']['mode'],
                                                 factor = cfg_content['scheduler']['factor'])
###################################### manuall modify###############################

#start training
for epoch in range(cfg_content['train']['max_epoch']) :

    train_acc = 0.0
    net.train()
    # train_loader 被分成batch_size大小的n各部分，step的最大值就是n。
    for step, data in enumerate (train_loader,start=0):

        images, labels = data
        str1 = images.size()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = net(images)
        loss = loss_function(logits,labels)
        predict = torch.max(logits, dim=1)[1]
        train_acc += (predict == labels).sum().item()
        loss.backward()
        optimizer.step()

        if(step % 10 == 9):

            iter = epoch*len(train_loader)+step
            #sys.stdout.flush()
            #print("[epoch %d] [batch count= %d], loss = %.4f" %(epoch, step, loss.item()))
            tempstr = "[epoch " + str(epoch) + "] [batch count= " + str(step) + "], loss = " + str(loss.item())
            logging.debug(tempstr)

            writer.add_scalar("Train/loss",loss.item(), iter)

    writer.add_scalar("Train/acc", train_acc/train_len, epoch)

    val_acc = 0.0
    val_loss = 0.0
    net.eval()
    with torch.no_grad():
        for step, data in enumerate(val_loader, start=0):
            val_images, val_labels = data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            output = net(val_images)
            val_loss += loss_function(output, val_labels)
            predict_y = torch.max(output, dim=1)[1]
            val_acc += (predict_y == val_labels.to(device)).sum().item()
            if(step % 10 == 9):
                iter = epoch*len(val_loader)+step
                writer.add_scalar("Val/loss", val_acc/10.0, iter)
                val_loss = 0.0

        avg_acc = val_acc / val_len
        scheduler.step(avg_acc)
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar("Train/lr", lr, epoch)
        writer.add_scalar("Val/acc", avg_acc, epoch)
        #sys.stdout.flush()
        #print("[epoch %d]  accurate = %.4f, learning rate = %.4g" %(epoch, avg_acc, lr ))
        tempstr = "[epoch " + str(epoch) + "]  accurate = " + str(avg_acc) + ", learning rate = " + str(lr)
        logging.debug(tempstr)

    if (epoch % 100 == 99):
        save_path1 = preprocess.save_mpdel_path + '/' + cfg_content['model']['name'] +'_'+ str(epoch+1)+'.pth'
        torch.save(net.state_dict(), save_path1)

print('Finished Training')
