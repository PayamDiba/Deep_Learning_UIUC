import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from DeepCNN import CNN
from IOtools import build_data_loader
from EvaluateCNN import evalPerformance, evalTestPerformance_MC

## Setup model
#TODO: read from check points
model = CNN()

train_data, test_data = build_data_loader('Data/', train_batch_size = 100, test_batch_size = 100)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
if device == 'cuda':
    # Pytorch will only use one GPU by default. The below command distributes computations on all GPUs
    model = nn.DataParallel(model)
    #The below code optimizes the algorithms for hardware. When having equal number of training data in
    #each iteration the below code often enhances the run time.
    cudnn.benchmark = True


## Train
for epoch in range(50):
    model.train()
    running_loss = 0

    for data in train_data:
        x,y = data

        x = x.to(device)
        y = y.to(device)

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

    if (epoch+1)%10 == 0:
        test_performance_MC = evalTestPerformance_MC(model,test_data, device)

    #TODO: Save Checkpoints

    print('Epoch: ' + str(epoch+1))
    print('Train Accuracy: ' + str(train_performance))
    print('Test Accuracy: ' + str(test_performance))
    if (epoch+1)%10 == 0:
        print('Test Accuracy using MC: ' + str(test_performance_MC))
