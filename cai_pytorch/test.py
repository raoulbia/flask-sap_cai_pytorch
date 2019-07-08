import torch
import torch.nn as nn
import os
from cai_pytorch.model import MyModel

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        # Computes the outputs / predictions
        return None #self.a + self.b * x


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    # checkpoint = torch.load(checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)


    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


# Now we can create a model and send it at once to the device
model = ManualLinearRegression()
# print(model.state_dict())

checkpoint = load_checkpoint(checkpoint='/home/vagrant/vmtest/cai_pytorch/cai_pytorch/static/4000_checkpoint.tar', model=model)
print(checkpoint.keys())
# print(model.parameters())
