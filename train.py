#-*-coding:utf-8-*-

import torch
import logging
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from configure import config as cfg
#sys.path.append("..")

from models import resnet
from data_processor import preprocess



logging.basicConfig(level=logging.DEBUG,
                    filename=cfg.log_path,
                    filemode='a',
                    format = '%(asctime)s - %(levelname)s:  %(message)s')

train_loader = preprocess.train_loader
val_loader = preprocess.val_loader
val_len = preprocess.val_len
writer = SummaryWriter(cfg.save_mpdel_path)
#data_root = os.getcwd()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# load model
net = resnet.resnet34(cfg.num_classes)
net.to(device)

#set optimizer
loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(),lr=cfg.lr)
optimizer = optim.SGD(net.parameters(),lr=cfg.lr, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.8)

#start training
for epoch in range(cfg.max_epoch) :

    net.train()
    # train_loader 被分成batch_size大小的n各部分，step的最大值就是n。
    for step, data in enumerate (train_loader,start=0):

        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = net(images)
        loss = loss_function(logits,labels)
        loss.backward()
        optimizer.step()

        if(step % 10 == 9):

            iter = epoch*len(train_loader)+step
            #sys.stdout.flush()
            #print("[epoch %d] [batch count= %d], loss = %.4f" %(epoch, step, loss.item()))
            tempstr = "[epoch " + str(epoch) + "] [batch count= " + str(step) + "], loss = " + str(loss.item())
            logging.debug(tempstr)

            writer.add_scalar("Train/loss",loss.item(), iter)

    train_accurate = 0.0
    acc = 0.0

    net.eval()
    with torch.no_grad():
        for step, data in enumerate(val_loader, start=0):
            val_images, val_labels = data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            output = net(val_images)
            predict_y = torch.max(output, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()

        avg_acc = acc / val_len
        scheduler.step(avg_acc)
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar("Train/lr", lr, epoch)
        writer.add_scalar("Val/acc", avg_acc, epoch)
        #sys.stdout.flush()
        #print("[epoch %d]  accurate = %.4f, learning rate = %.4g" %(epoch, avg_acc, lr ))
        tempstr = "[epoch " + str(epoch) + "]  accurate = " + str(avg_acc) + ", learning rate = " + str(lr)
        logging.debug(tempstr)

    save_path1 = cfg.save_mpdel_path + '/resnet34_'+ str(epoch)+'.pth'
    torch.save(net.state_dict(), save_path1)

print('Finished Training')
