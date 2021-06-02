import os
import torch 
import argparse
import yaml
import time
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from datasets import augmentations

from utils.loss import OhemCELoss
from utils.scheduler import WarmupPolyLR
from models import choose_models
from datasets import choose_datasets
from datasets.augmentations import Compose, randomCrop, randomHorizontalFlip
from utils.utils import fix_seeds, setup_cuda


def main(cfg):
    start = time.time()
    if not os.path.exists(cfg['TRAIN']['OUTPUT']): os.makedirs(cfg['TRAIN']['OUTPUT'])

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cuda()

    augmentations = Compose([
        randomHorizontalFlip(p=0.3),
        randomCrop((480, 640), p=0.2)
    ])

    device = torch.device(cfg['MODEL']['DEVICE'])

    train_dataset = choose_datasets[cfg['DATASET']['NAME']](root=cfg['DATASET']['ROOT'], mode='train', img_size=cfg['TRAIN']['IMG_SIZE'], augmentations=augmentations)
    train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True)

    test_dataset = choose_datasets[cfg['DATASET']['NAME']](root=cfg['DATASET']['ROOT'], mode='val', img_size=cfg['EVAL']['IMG_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    model = choose_models[cfg['MODEL']['NAME']](train_dataset.n_classes)
    model.init_weights(cfg['MODEL']['PRETRAINED'])
    model = model.to(device)

    main_criteria = OhemCELoss(thresh=0.7, ignore_label=train_dataset.ignore_label)
    aux_criteria = [OhemCELoss(thresh=0.7, ignore_label=train_dataset.ignore_label) for _ in range(model.num_aux_heads)]

    epochs = cfg['TRAIN']['EPOCHS']
    n_classes = cfg['DATASET']['NUM_CLASSES']
    best_mIoU = 0
    iters_per_epoch = int(len(train_dataset) / cfg['TRAIN']['BATCH_SIZE'])
    max_iters = epochs * iters_per_epoch
    warmup_iters = iters_per_epoch * cfg['TRAIN']['WARMUP']

    optimizer = SGD(model.parameters(), lr=cfg['TRAIN']['LR'], momentum=0.9, weight_decay=cfg['TRAIN']['DECAY'])
    scheduler = WarmupPolyLR(optimizer, power=0.9, max_iter=max_iters, warmup_iter=warmup_iters, warmup_ratio=0.1)

    writer = SummaryWriter(os.path.join(cfg['TRAIN']['OUTPUT'], 'logs'))

    for epoch in range(1, epochs+1):
        model.train()

        total_loss = []
        pbar = tqdm(enumerate(train_loader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")

        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)

            main_logits, *aux_logits = model(img)
            main_loss = main_criteria(main_logits, lbl)
            aux_loss = [crit(logits, lbl) for crit, logits in zip(aux_criteria, aux_logits)]
            loss = main_loss + sum(aux_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            scheduler.step()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            total_loss.append(loss.item())

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")
        
        train_loss_per_epoch = sum(total_loss) / len(total_loss)
        writer.add_scalar('train/loss', train_loss_per_epoch, epoch)
        writer.add_scalar('train/lr', lr, epoch)

        torch.save(model.state_dict(), os.path.join(cfg['TRAIN']['OUTPUT'], 'final.pth'))
        torch.cuda.empty_cache()
       
        if (epoch % cfg['EVAL']['INTERVAL'] == 0) and (epoch >= cfg['EVAL']['INTERVAL']):
            print('Evaluating...')
            model.eval()

            hist = torch.zeros(n_classes, n_classes).to(device)

            with torch.no_grad():
                for images, labels in tqdm(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    preds = model(images)
                    keep = labels != cfg['DATASET']['IGNORE_LABEL']
                    hist += torch.bincount(labels[keep] * n_classes + preds[keep], minlength=n_classes**2).view(n_classes, n_classes)
        
            ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
            mIoU = ious[~ious.isnan()].mean().item()

            writer.add_scalar('val/mIoU', mIoU, epoch)

            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save(model.state_dict(), os.path.join(cfg['TRAIN']['OUTPUT'], 'best.pth'))

            print(f"mIoU: {mIoU:4.4f} Best mIoU: {best_mIoU:4.4f}")

    writer.close()

    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = [
        ['Best mIoU', best_mIoU],
        ['Final Train Loss', train_loss_per_epoch],
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