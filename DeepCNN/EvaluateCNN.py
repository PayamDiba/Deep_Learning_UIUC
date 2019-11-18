import torch

def evalPerformance(model, dataloader, device):
    """Evaluate the performance of model on provided data.
    It evokes the eval() mode to turn of all dropout layers.

    Args:
        dataloader (torch.utils.data.DataLoader) either of train or test dataloaders
        device: 'cuda' or 'cpu'
    Returns:
        ratio of correctly predicted labels
    """
    model = model.to(device)
    model.eval()

    nData = 0
    nCorrect = 0

    with torch.no_grad():
        for data in dataloader:
            x,y = data
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            _, y_hat = torch.max(out,1)

            nData += y.shape[0]
            nCorrect += (y_hat == y).sum().item()

        return nCorrect/nData


def evalTestPerformance_MC(model, dataloader, device):
    """Evaluate the performance of model on provided data.
    It uses Monte Carlo method. So, dropout layers are kept as they are in the trainig
    but predictions are averaged over 100 evaluations.

    Note: don't run this on Train set because train dataloader is shuffled after
    each epoch.

    Args:
        dataloader (torch.utils.data.DataLoader) either of train or test dataloaders
        device: 'cuda' or 'cpu'
    Returns:
        ratio of correctly predicted labels
    """
    model = model.to(device)
    model.train()

    nData = 0
    nCorrect = 0


    with torch.no_grad():
        for data in dataloader:
            x,y = data
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            for i in range(49):
                out += model(x)
            ## There is no need to divide by total number, since we get the max anyway
            _, y_hat = torch.max(out,1)

            nData += y.shape[0]
            nCorrect += (y_hat == y).sum().item()

        return nCorrect/nData
