import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from pprint import pprint
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


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    
    traintransform = get_train_transform(train_cfg['IMAGE_SIZE'], seg_fill=0)
    valtransform = get_val_transform(eval_cfg['IMAGE_SIZE'])

    trainset = get_dataset(dataset_cfg['NAME'], dataset_cfg['ROOT'], 'train', traintransform)
    valset = get_dataset(dataset_cfg['NAME'], dataset_cfg['ROOT'], 'val', valtransform)
    
    trainsampler, valsampler = get_sampler(trainset, valset, train_cfg['DDP'])
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=trainsampler)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, sampler=valsampler)

    model = get_model(model_cfg['NAME'], model_cfg['VARIANT'], trainset.n_classes)
    model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)
    if train_cfg['DDP']: model = DDP(model, device_ids=[gpu])

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None, loss_cfg['THRESH'])
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()

        if train_cfg['DDP']: trainsampler.set_epoch(epoch)

        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)
            optimizer.zero_grad()

            with autocast(enabled=train_cfg['AMP']):
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

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['VARIANT']}_{dataset_cfg['NAME']}_final.pth")
       
        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            miou = evaluate(model, valloader, device)[-1]
            writer.add_scalar('val/mIoU', miou, epoch)

            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['VARIANT']}_{dataset_cfg['NAME']}.pth")
            print(f"Current mIoU: {miou:4.4f} Best mIoU: {best_mIoU:4.4f}")

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
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    pprint(cfg)
    fix_seeds(123)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()