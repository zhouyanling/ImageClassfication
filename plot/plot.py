#-*-coding:utf-8-*-#-*-coding:utf-8-*-
import matplotlib.pyplot as plt
import sys
import numpy as np
input = open(sys.argv[1])

loss_list = []
acc_list = []

for line in input:

    #get loss value
    if(line.rfind(" loss = ") != -1):
        loss_index = line.rindex(" loss = ")
        sig_index = line.rindex(",")
        loss_val = line[loss_index + 8: sig_index]
        loss_list.append(float(loss_val))
    else:
        continue

    #get accurate value
    if(line.rfind("accurate = ") != -1):
        acc_index = line.rindex(" ")
        acc_val = line[acc_index+1 : acc_index +5]
        acc_list.append(float(acc_val))
    else:
        continue

loss_len = len(loss_list)
acc_len = len(acc_list)


loss_x = np.arange(0, loss_len, 1)
acc_x = np.arange(0, acc_len, 1)

#y1 = loss_list
#y2 = acc_list

plt.figure("subplot")

plt.subplot(211)
plt.plot(loss_x,loss_list)
#plt.xticks(np.arange(0,300,10))
#plt.yticks(np.arange(0,4,1))

plt.subplot(212)
plt.plot(acc_x, acc_list,'r--')
#plt.xticks(np.arange(0,300,10))
#plt.yticks(np.arange(0,1,0.1))
#plt.legend()
plt.show()