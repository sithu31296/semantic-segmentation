import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        if preds.shape[2:] != labels.shape[1:]:
            preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)
        return self.criterion(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        if preds.shape[2:] != labels.shape[1:]:
            preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)

        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and targets in shape [B, C, H, W]
        if preds.shape[2:] != targets.shape[2:]:
            preds = F.interpolate(preds, size=targets.shape[2:], mode='bilinear', align_corners=False)

        tp = torch.sum(targets*preds, dim=(2, 3))
        fn = torch.sum(targets*(1-preds), dim=(2, 3))
        fp = torch.sum((1-targets)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1-dice_score, dim=-1)

        # adjust loss to account for number of classes
        dice_score = dice_score / targets.shape[1]
        return dice_score.mean()



__all__ = ['ce', 'ohemce', 'dice']

def get_loss(loss_fn_name: str = 'ce', ignore_label: int = 255, cls_weights: Tensor = None, thresh: float = 0.7):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'ohemce':
        return OhemCrossEntropy(ignore_label, cls_weights, thresh)
    elif loss_fn_name == 'dice':
        return Dice(thresh)
    return CrossEntropy(ignore_label, cls_weights)



if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 224, 224), dtype=torch.float)
    label = torch.randint(0, 19, (2, 224, 224), dtype=torch.long)
    loss_fn = OhemCrossEntropy(0.7)
    y = loss_fn(pred, label)
    print(y)