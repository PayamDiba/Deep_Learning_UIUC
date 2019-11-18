import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from DeepResNet import ResNet
from IOtools import build_CIFAR100_DataLoader, build_TinyImageNet_DataLoader
from torch.autograd import Variable
from EvaluateNN import evalPerformance
from collections import OrderedDict

## Setup model
#TODO: read from check points
model = ResNet(output_size = 200)

#Adjust the model structure for input size 64*64
model.maxPool = nn.MaxPool2d(kernel_size=4, stride=4)
model.fc = nn.Sequential(OrderedDict([
    ('dropout_fc', nn.Dropout(p = 0.5)),
    ('fc1', nn.Linear(model.fc.in_features*4, model.fc.in_features)),
    ('fc2', nn.Linear(model.fc.in_features, 200))
]))



train_path = '/u/training/tra179/scratch/HW4/Data/tiny-imagenet-200/train'
val_path = '/u/training/tra179/scratch/HW4/Data/tiny-imagenet-200/val/images'
train_data, test_data = build_TinyImageNet_DataLoader(train_path = train_path, val_path = val_path, train_batch_size = 100, val_batch_size = 100, num_workers = 2)
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
        # ##### Specific to Blue Waters #####
        if(epoch>5):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'step' in state.keys():
                        if(state['step']>=1024):
                            state['step'] = 1000
        # ###################################
        optimizer.step()

    train_performance = evalPerformance(model,train_data, device)
    test_performance = evalPerformance(model,test_data, device)

    #TODO: Save Checkpoints

    print('Epoch: ' + str(epoch+1))
    print('Train Accuracy: ' + str(train_performance))
    print('Test Accuracy: ' + str(test_performance))
