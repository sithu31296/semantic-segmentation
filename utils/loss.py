import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_lb=255) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def forward(self, pred, labels):
        return self.criterion(pred, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, thresh, ignore_label=255) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')

    def forward(self, pred, labels):
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(pred, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)


class Dice(nn.Module):
    def __init__(self, delta=0.5, smooth=1e-6):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        smooth: Smoothening constant to prevent division by zero errors
        """
        super().__init__()
        self.delta = delta
        self.smooth = smooth

    def forward(self, preds, targets):
        tp = torch.sum(targets*preds, dim=(2, 3))
        fn = torch.sum(targets*(1-preds), dim=(2, 3))
        fp = torch.sum((1-targets)*preds, dim=(2, 3))

        dice_score = (tp + self.smooth)/(tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        dice_score = torch.sum(1-dice_score, dim=-1)

        # adjust loss to account for number of classes
        dice_score = dice_score / targets.shape[1]

        return dice_score

# class Focal(nn.Module):
#     def __init__(self, alpha=None, beta=None, gamma=2.):
#         """
#         alpha: Controls weight given to each class
#         beta: Controls relative weight of FP and FN. beta > 0.5 penalises FN more than FP
#         gamma: Focal parameter controls degree of down-weighting of easy examples
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma

#     def forward(self, preds, targets):
        