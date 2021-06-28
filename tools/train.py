import torch 
import argparse
import yaml
import time
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import SGD
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.insert(0, '.')
from utils.loss import OhemCELoss
from utils.scheduler import WarmupPolyLR
from models import choose_models
from datasets import choose_datasets
from datasets.augmentations import Compose, randomCrop, randomHorizontalFlip
from utils.utils import fix_seeds, setup_cudnn
from val import evaluate


def main(cfg):
    start = time.time()
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cudnn()

    transforms = Compose([
        randomHorizontalFlip(p=0.3),
        randomCrop((480, 640), p=0.2)
    ])

    device = torch.device(cfg['DEVICE'])

    train_dataset = choose_datasets(cfg['DATASET']['NAME'], cfg['DATASET']['ROOT'], 'train', cfg['TRAIN']['IMG_SIZE'], transforms)
    train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True)

    test_dataset = choose_datasets(cfg['DATASET']['NAME'], cfg['DATASET']['ROOT'], 'val', cfg['EVAL']['IMG_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    model = choose_models(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], train_dataset.n_classes, cfg['TRAIN']['IMG_SIZE'][0])
    model.init_weights(cfg['MODEL']['PRETRAINED'])
    model = model.to(device)

    criterion = OhemCELoss(thresh=0.7, ignore_label=train_dataset.ignore_label)

    epochs = cfg['TRAIN']['EPOCHS']
    best_mIoU = 0
    iters_per_epoch = int(len(train_dataset) / cfg['TRAIN']['BATCH_SIZE'])
    max_iters = epochs * iters_per_epoch
    warmup_iters = iters_per_epoch * cfg['TRAIN']['WARMUP']

    optimizer = SGD(model.parameters(), lr=cfg['TRAIN']['LR'], momentum=0.9, weight_decay=cfg['TRAIN']['DECAY'])
    scheduler = WarmupPolyLR(optimizer, power=0.9, max_iter=max_iters, warmup_iter=warmup_iters, warmup_ratio=0.1)
    scaler = GradScaler(enabled=cfg['TRAIN']['AMP'])

    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(1, epochs+1):
        model.train()

        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")

        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)

            with autocast(enabled=cfg['TRAIN']['AMP']):
                logits = model(img)
                logits = F.interpolate(logits, size=lbl.shape[-2:], mode='bilinear', align_corners=True)
                loss = criterion(logits, lbl)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")
        
        train_loss /= len(train_dataset)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.flush()
       
        if (epoch % cfg['EVAL_INTERVAL'] == 0) and (epoch >= cfg['EVAL_INTERVAL']):
            _, miou = evaluate(model, test_loader, device)

            writer.add_scalar('val/mIoU', miou, epoch)
            writer.flush()

            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}_{cfg['DATASET']['NAME']}.pth")

            print(f"mIoU: {miou:4.4f} Best mIoU: {best_mIoU:4.4f}")

    writer.close()
    pbar.close()

    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = [
        ['Best mIoU', best_mIoU],
        ['Total Training Time', total_time]
    ]
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/camvid.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)