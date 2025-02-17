import torch
import torch.nn as nn
import torch.nn.functional as F

def _one_hot(label, num_classes, device, dtype, requires_grad=True):
    """Converts a label tensor to a one-hot tensor."""
    one_hot = torch.eye(num_classes, device=device, requires_grad=requires_grad, dtype=dtype)[label.squeeze(1)]
    return one_hot.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

class DiceLoss(nn.Module):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps
    
    def forward(self, logits, true):
        num_classes = logits.shape[1]
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)
    
class JaccardLoss(nn.Module):
    """Jaccard (IoU) loss."""

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        true_1_hot = _one_hot(target, num_classes, logits.device, logits.dtype, requires_grad=False)
        probas = F.softmax(logits, dim=1)
        dims = (0,) + tuple(range(2, target.ndim))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jaccard_loss = (intersection / (union + self.eps)).mean()
        return 1. - jaccard_loss

class FocalLoss(nn.Module):
    """
    Focal Loss.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        true_1_hot = _one_hot(target, num_classes, logits.device, logits.dtype, requires_grad=False)
        probas = F.softmax(logits, dim=1)  # Probabilities
        ce_loss = F.cross_entropy(logits, target.squeeze(1), reduction="none")

        if self.alpha is not None:
            if isinstance(self.alpha, list):
                alpha = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            else: #Should be a tensor
                alpha = self.alpha
            alpha = alpha.gather(0, target.view(-1)) #Pick the appropriate class weight
            ce_loss = alpha * ce_loss

        pt = (true_1_hot * probas).sum(dim=1) #Probability of the true class
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss  # (N, H, W)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")