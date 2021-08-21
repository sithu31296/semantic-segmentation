import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F

import sys
sys.path.insert(0, '.')
from utils.utils import setup_cudnn
from datasets import get_dataset
from datasets.transforms import get_val_transform
from models import get_model


@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()

    n_classes = dataloader.dataset.n_classes
    ignore_label = dataloader.dataset.ignore_label
    hist = torch.zeros(n_classes, n_classes).to(device)

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=True)
        preds = logits.softmax(dim=1).argmax(dim=1)

        keep = labels != ignore_label
        hist += torch.bincount(labels[keep] * n_classes + preds[keep], minlength=n_classes**2).view(n_classes, n_classes)
    
    ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
    miou = ious[~ious.isnan()].mean().item()

    return ious.cpu().numpy().tolist(), miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    ignore_label = dataloader.dataset.ignore_label
    hist = torch.zeros(n_classes, n_classes).to(device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        preds = scaled_logits.argmax(dim=1)
        keep = labels != ignore_label
        hist += torch.bincount(labels[keep] * n_classes + preds[keep], minlength=n_classes**2).view(n_classes, n_classes)
    
    ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
    miou = ious[~ious.isnan()].mean().item()

    return ious.cpu().numpy().tolist(), miou * 100


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_transform(eval_cfg['IMAGE_SIZE'])
    dataset = get_dataset(cfg['DATASET']['NAME'], cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}_{cfg['DATASET']['NAME']}.pth"

    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    if eval_cfg['MSF']['ENABLE']:
        ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
    else:
        ious, miou = evaluate(model, dataloader, device)

    table = {
        'Class': list(dataset.CLASSES),
        'IoU': ious
    }

    print(tabulate(table, headers='keys'))
    print(f"\nOverall mIoU: {miou:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/camvid.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)