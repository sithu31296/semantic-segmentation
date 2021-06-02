import os
import torch
import argparse
import yaml

from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader

from utils.utils import setup_cuda
from datasets import choose_datasets
from models import choose_models


@torch.no_grad()
def evaluate(cfg, model, dataloader, device):
    model.eval()

    n_classes = cfg['DATASET']['NUM_CLASSES']
    hist = torch.zeros(n_classes, n_classes).to(device)

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        keep = labels != cfg['DATASET']['IGNORE_LABEL']

        hist += torch.bincount(labels[keep] * n_classes + preds[keep], minlength=n_classes**2).view(n_classes, n_classes)
    
    ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
    miou = ious[~ious.isnan()].mean().item()

    return ious.cpu().numpy().tolist(), miou


def main(cfg):
    setup_cuda()

    device = torch.device(cfg['MODEL']['DEVICE'])

    test_dataset = choose_datasets[cfg['DATASET']['NAME']](root=cfg['DATASET']['ROOT'], mode='val', img_size=cfg['EVAL']['IMG_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    model = choose_models[cfg['MODEL']['NAME']](cfg['DATASET']['NUM_CLASSES'])
    model.load_state_dict(torch.load(cfg['TEST']['TRAINED_MODEL'], map_location='cpu'))
    model = model.to(device)

    ious, miou = evaluate(cfg, model, test_loader, device)

    table = {
        'Class': list(test_dataset.class_names),
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