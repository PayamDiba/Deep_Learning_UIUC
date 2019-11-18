import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from critic import critic
from generator import generator
from data_tools import build_data_loader
from utils import calc_gradient_penalty, saveGenOut
from EvaluateCNN import evalPerformanceCritic
import time

"""
make a random batch of noise, with batch size of 100
"""
n_z = 100 #input dimension to generator
n_classes = 10 #number of classes
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()


"""
Training
"""
num_epochs = 200
batch_size = 128
gen_train = 1


train_data, test_data = build_data_loader('Data/', train_batch_size = batch_size, test_batch_size = batch_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## setup model
aD = critic()
aD.to(device)

aG = generator()
aG.to(device)

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))
criterion = nn.CrossEntropyLoss()

##### Print losses to monitor #####
loss1 = []
loss2 = []
loss3 = []
loss4 = []
loss5 = []
acc1 = []
###################################


start_time = time.time()
for epoch in range(0,num_epochs):

    for group in optimizer_d.param_groups:
        for p in group['params']:
            state = optimizer_d.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000

    for group in optimizer_g.param_groups:
        for p in group['params']:
            state = optimizer_g.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000

    aG.train()
    aD.train()

    for batch_idx, data in enumerate(train_data):
        x,y = data
        if(y.shape[0] < batch_size):
            continue

        """
        train G
        """
        if batch_idx % gen_train == 0:
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            optimizer_g.step()

        """
        train D
        """
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # calculate discriminator loss with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)

        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)

        # calculate discriminator loss with real data
        real_data = Variable(x).cuda()
        real_label = Variable(y).cuda()

        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data, batch_size)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()

        optimizer_d.step()

        """
        Append losses and print
        """
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if batch_idx % 50 == 0:
            print(epoch, batch_idx, "%.2f" % np.mean(loss1),
                                    "%.2f" % np.mean(loss2),
                                    "%.2f" % np.mean(loss3),
                                    "%.2f" % np.mean(loss4),
                                    "%.2f" % np.mean(loss5),
                                    "%.2f" % np.mean(acc1))

    test_performance = evalPerformanceCritic(aD,test_data, device)
    print('Testing',test_performance, time.time()-start_time)

    if (epoch+1) % 1 == 0:
        torch.save(aG,'tempG.model')
        torch.save(aD,'tempD.model')
    """
    Evaluate Generator and save its output
    """
    saveGenOut(aG, save_noise, epoch, device)
torch.save(aG,'generator.model')
torch.save(aD,'discriminator.model')
