import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from DeepResNet import ResNet
from IOtools import build_CIFAR100_DataLoader, build_TinyImageNet_DataLoader
from torch.autograd import Variable
from EvaluateNN import evalPerformance
import torch.distributed as dist

import os
import subprocess
from mpi4py import MPI
import numpy as np



#### Initilize distributed training

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor
############################################

def average_gradients(model):
    """
    This function averages gradients over all GPU nodes. On Blue Waters (I believe because of MPI backend) we need
    to transfer data back to CPU to perform comuptations and then send it back again
    to GPU.
    """
    for param in model.parameters():
        #print(param.grad.data)
        tensor0 = param.grad.data.cpu()
        dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
        tensor0 /= float(num_nodes)
        param.grad.data = tensor0.cuda()


## Setup model
#TODO: read from check points
model = ResNet(output_size = 100)

#Make sure that all nodes have the same model (intial parameters)
for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))

train_data, test_data = build_CIFAR100_DataLoader('Data/', train_batch_size = 100, test_batch_size = 100, num_workers = 0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# adjust to Pytorch 0.3.0
if device == 'cuda':
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
if device == 'cuda':
    #The below code optimizes the algorithms for hardware. When having equal number of training data in
    #each iteration the below code often enhances the run time.
    cudnn.benchmark = True

## Train
for epoch in range(50):
    model.train()
    running_loss = 0

    for data in train_data:
        x,y = data

        if device == 'cuda':
            x = x.cuda()
            y = y.cuda()

        x, y = Variable(x), Variable(y) # Guess it is needed in Pytorch 0.3.0

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y.long())
        loss.backward()
        average_gradients(model)
        ##### Specific to Blue Waters #####
        if(epoch>6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'step' in state.keys():
                        if(state['step']>=1024):
                            state['step'] = 1000
        ###################################
        optimizer.step()

    train_performance = evalPerformance(model,train_data, device)
    test_performance = evalPerformance(model,test_data, device)

    #TODO: Save Checkpoints

    #Note: different nodes give slightly different results because of the batch-nornalization
    #Since each node recieves a different batch and therefore it has different statistics (mean and variance) for batch-norm
    #Only print the results of the first node.
    if rank == 0:
        print('Epoch: ' + str(epoch+1))
        print('Train Accuracy: ' + str(train_performance))
        print('Test Accuracy: ' + str(test_performance))
