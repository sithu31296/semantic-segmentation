import torch 
import argparse
import yaml
import time
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, '.')
from utils.losses import get_loss
from utils.schedulers import get_scheduler
from utils.optimizers import get_optimizer
from models import get_model
from datasets import get_dataset
from datasets.transforms import get_train_transform, get_val_transform
from datasets.samplers import get_sampler
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate


def main(cfg):
    start = time.time()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)

    device = torch.device(cfg['DEVICE'])
    ddp_enable = cfg['TRAIN']['DDP']
    epochs = cfg['TRAIN']['EPOCHS']
    best_mIoU = 0
    gpu = setup_ddp()
    
    valtransform = get_val_transform(cfg['EVAL']['IMAGE_SIZE'])
    valset = get_dataset(cfg['DATASET']['NAME'], cfg['DATASET']['ROOT'], 'val', valtransform)

    traintransform = get_train_transform(cfg['TRAIN']['IMAGE_SIZE'], seg_fill=0)
    trainset = get_dataset(cfg['DATASET']['NAME'], cfg['DATASET']['ROOT'], 'train', traintransform)

    trainsampler, valsampler = get_sampler(trainset, valset, ddp_enable)
    trainloader = DataLoader(trainset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True, sampler=trainsampler)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, sampler=valsampler)

    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], trainset.n_classes)
    model.init_weights(cfg['MODEL']['PRETRAINED'])
    model = model.to(device)
    if ddp_enable: model = DDP(model, device_ids=[gpu])

    iters_per_epoch = int(len(trainset) / cfg['TRAIN']['BATCH_SIZE'])
    loss_fn = get_loss(cfg['LOSS']['NAME'], trainset.ignore_label, None, cfg['LOSS']['THRESH'])
    optimizer = get_optimizer(model, cfg['TRAIN']['OPTIMIZER']['NAME'], cfg['TRAIN']['OPTIMIZER']['LR'], cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY'])
    scheduler = get_scheduler(cfg['TRAIN']['SCHEDULER']['NAME'], optimizer, epochs * iters_per_epoch, cfg['TRAIN']['SCHEDULER']['POWER'], iters_per_epoch * cfg['TRAIN']['SCHEDULER']['WARMUP'], cfg['TRAIN']['SCHEDULER']['WARMUP_RATIO'])
    scaler = GradScaler(enabled=cfg['TRAIN']['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()

        if ddp_enable: trainsampler.set_epoch(epoch)

        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")

        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)
            optimizer.zero_grad()

            with autocast(enabled=cfg['TRAIN']['AMP']):
                logits = model(img)
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")
        
        train_loss /= iter
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.flush()
        torch.cuda.empty_cache()
       
        if (epoch % cfg['TRAIN']['EVAL_INTERVAL'] == 0) and (epoch >= cfg['TRAIN']['EVAL_INTERVAL']):
            _, miou = evaluate(model, valloader, device)

            writer.add_scalar('val/mIoU', miou, epoch)
            writer.flush()

            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.module.state_dict() if ddp_enable else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}_{cfg['DATASET']['NAME']}.pth")
            print(f"mIoU: {miou:4.4f} Best mIoU: {best_mIoU:4.4f}")

    writer.close()
    pbar.close()
    cleanup_ddp()

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

    fix_seeds(123)
    setup_cudnn()
    main(cfg)