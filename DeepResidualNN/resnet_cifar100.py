import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from DeepResNet import ResNet
from IOtools import build_CIFAR100_DataLoader, build_TinyImageNet_DataLoader
from torch.autograd import Variable
from EvaluateNN import evalPerformance

## Setup model
#TODO: read from check points
model = ResNet(output_size = 100)

train_data, test_data = build_CIFAR100_DataLoader('Data/', train_batch_size = 100, test_batch_size = 100, num_workers = 2)
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

    print('Epoch: ' + str(epoch+1))
    print('Train Accuracy: ' + str(train_performance))
    print('Test Accuracy: ' + str(test_performance))
