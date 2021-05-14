import torch


class FlippedCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, period=100, eta_max=1e-2, last_epoch=-1):
        import numpy as np
        lr = optimizer.param_groups[0]['lr']
        if lr > eta_max:
            raise

        def flipped_cosine(step):
            coeff = (0.5 - np.cos(step * (2 * np.pi) / period) / 2.) * (eta_max - lr)/lr + 1.
            return coeff

        super(FlippedCosineSchedule, self).__init__(optimizer,
                                                    flipped_cosine,
                                                    last_epoch=last_epoch)


if __name__ == '__main__':
    class ToyModel(torch.nn.Module):
        def __init__(self, hidden_size):
            super(ToyModel, self).__init__()
            self.linear_1 = torch.nn.Linear(1, hidden_size)
            self.linear_2 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_3 = torch.nn.Linear(hidden_size, 1)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.linear_1(x))
            x = self.relu(self.linear_2(x))
            x = self.sigmoid(self.linear_3(x))
            return x

    model = ToyModel(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = FlippedCosineSchedule(optimizer, period=100, eta_max=1e-2)
    for step in range(100):
        scheduler.step()
