import torch

def evalPerformanceCritic(model, dataloader, device):
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

            _,out = model(x)
            _, y_hat = torch.max(out,1)

            nData += y.shape[0]
            nCorrect += (y_hat == y).sum().item()

        return nCorrect/nData
