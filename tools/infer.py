import torch
import argparse
import yaml
from torch.nn import functional as F
from pathlib import Path
from torchvision import io

import sys
sys.path.insert(0, '.')
from models import choose_models
from datasets import choose_datasets
from utils.utils import time_synschronized


class Model:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['DEVICE'])
        self.dataset = choose_datasets(cfg['DATASET']['NAME'], cfg['DATASET']['ROOT'], 'test', cfg['TEST']['IMG_SIZE'])
        self.model = choose_models(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], self.dataset.n_classes, cfg['TEST']['IMG_SIZE'][0])
        self.model.load_state_dict(torch.load(cfg['TRAINED_MODEL'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def predict(self, image):
        img = image.float()
        img /= 255
        img = self.dataset.image_transforms(img).unsqueeze(0).to(self.device)
        start = time_synschronized()
        pred = self.model(img)
        end = time_synschronized()
        print(f"Model Inference Time: {(end-start)*1000}ms")

        pred = F.interpolate(pred, size=img.shape[2:], mode='bilinear', align_corners=False)
        # pred = F.softmax(pred, dim=1)
        pred = pred.argmax(dim=1)
        pred = self.dataset.decode(pred).cpu().squeeze().permute(2, 0, 1).type(torch.uint8)
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/cityscapes.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    if not save_dir.exists(): save_dir.mkdir()
    
    model = Model(cfg)

    if cfg['TEST']['MODE'] == 'image':
        
        test_file = Path(cfg['TEST']['FILE'])
        if test_file.is_file():
            print(f'Inferencing {test_file}')
            image = io.read_image(str(test_file))
            segmap = model.predict(image)
            io.write_png(segmap, str(save_dir / f"{str(test_file.stem)}.png"))
        
        else:
            files = test_file.glob('*.*')
            for file in files:
                print(f'Inferencing {file}')
                image = io.read_image(str(file))
                segmap = model.predict(image)
                io.write_png(segmap, str(save_dir / f"{str(file.stem)}.png"))

    elif cfg['TEST']['MODE'] == 'video':
        pass
    else:
        pass

