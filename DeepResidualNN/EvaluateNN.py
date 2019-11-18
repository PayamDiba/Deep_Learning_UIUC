import torch
from torch.autograd import Variable

def evalPerformance(model, dataloader, device):
    """Evaluate the performance of model on provided data.
    It evokes the eval() mode to turn of all dropout layers.

    Args:
        dataloader (torch.utils.data.DataLoader) either of train or test dataloaders
        device: 'cuda' or 'cpu'
    Returns:
        ratio of correctly predicted labels
    """
    if device == 'cuda':
        model.cuda()

    model.eval()

    nData = 0
    nCorrect = 0

    # torch.no_grad() is not supported in pytorch 0.3.0
    #with torch.no_grad():

    for data in dataloader:
        x,y = data
        if device == 'cuda':
            x = x.cuda()
            y = y.cuda()

        x, y = Variable(x), Variable(y) # Guess it is needed in Pytorch 0.3.0
        out = model(x)
        _, y_hat = torch.max(out,1)

        nData += y.shape[0]
        nCorrect += (y_hat == y).sum().data[0]

    return nCorrect/nData

def evalPerformance_preTrained(model, dataloader, device, upsampler):
    """Evaluate the performance of model on provided data.
    It evokes the eval() mode to turn of all dropout layers.
    It applies the provide upsampler onto the input images

    Args:
        dataloader (torch.utils.data.DataLoader) either of train or test dataloaders
        device: 'cuda' or 'cpu'
    Returns:
        ratio of correctly predicted labels
    """
    if device == 'cuda':
        model.cuda()

    model.eval()

    nData = 0
    nCorrect = 0

    # torch.no_grad() is not supported in pytorch 0.3.0
    #with torch.no_grad():

    for data in dataloader:
        x,y = data
        x = upsampler(x)

        if device == 'cuda':
            x = x.cuda()
            y = y.cuda()

        #y = Variable(y) # Guess it is needed in Pytorch 0.3.0
        out = model(x)
        _, y_hat = torch.max(out,1)

        nData += y.shape[0]
        nCorrect += (y_hat == y).sum().item()

    return nCorrect/nData
