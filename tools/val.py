import torch
import argparse
import yaml

from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F

import sys
sys.path.insert(0, '.')
from utils.utils import fix_seeds, setup_cudnn
from datasets import choose_datasets
from models import choose_models


@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()

    n_classes = dataloader.dataset.n_classes
    hist = torch.zeros(n_classes, n_classes).to(device)

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=True)
        preds = torch.softmax(logits, dim=1)
        keep = labels != dataloader.dataset.ignore_label

        hist += torch.bincount(labels[keep] * n_classes + preds[keep], minlength=n_classes**2).view(n_classes, n_classes)
    
    ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
    miou = ious[~ious.isnan()].mean().item()

    return ious.cpu().numpy().tolist(), miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    hist = torch.zeros(n_classes, n_classes).to(device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        scaled_logits = torch.zeros_like(labels)

        for scale in scales:
            new_H, new_W = int(scale * labels.shape[2]), int(scale * labels.shape[3])
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)

            logits = model(scaled_images)
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=True)
            scaled_logits += torch.softmax(logits, dim=1)

            scaled_images = torch.flip(scaled_images, dims=(3,))
            logits = model(scaled_images)
            logits = torch.flip(logits, dims=(3,))
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=True)
            scaled_logits += torch.softmax(logits, dim=1)

        preds = scaled_logits.argmax(dim=1)
        keep = labels != dataloader.dataset.ignore_label

        hist += torch.bincount(labels[keep] * n_classes + preds[keep], minlength=n_classes**2).view(n_classes, n_classes)
    
    ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
    miou = ious[~ious.isnan()].mean().item()

    return ious.cpu().numpy().tolist(), miou


def main(cfg):
    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cudnn()

    device = torch.device(cfg['DEVICE'])

    test_dataset = choose_datasets(cfg['DATASET']['NAME'], root=cfg['DATASET']['ROOT'], split='val', img_size=cfg['EVAL']['IMG_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    model = choose_models(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], test_dataset.n_classes, cfg['EVAL']['IMG_SIZE'])
    model.load_state_dict(torch.load(cfg['TRAINED_MODEL'], map_location='cpu'))
    model = model.to(device)

    if cfg['EVAL']['MSF']:
        ious, miou = evaluate_msf(model, test_loader, device, cfg['EVAL']['SCALES'])
    else:
        ious, miou = evaluate(model, test_loader, device)

    table = {
        'Class': list(test_dataset.CLASSES),
        'IoU': ious
    }

    print(tabulate(table, headers='keys'))
    print(f"\nOverall mIoU: {miou:4.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/camvid.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)