
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: int=1, reduction: str="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        assert (self.reduction == "mean" or self.reduction == "sum" or self.reduction == "none"), "reduce argument must be: 'none', 'mean' or 'sum'"

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):
        cross_entropy_loss = F.cross_entropy(y_pred, y_true, reduction="none")
        probablilites = torch.exp(-cross_entropy_loss)
        focal_loss = torch.pow((1 - probablilites), self.gamma) * cross_entropy_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        elif self.reduction == "none":
            return focal_loss

        return focal_loss




def test_focal_loss():
    f = FocalLoss(reduction="mean")
    loss = f.forward(torch.tensor([[2.4, 0.32, 1.19, 3.3], [2.4, 0.32, 1.19, 3.3], [2.4, 0.32, 1.19, 3.3]]), torch.tensor([2, 1, 0]))
    print(loss)

    
