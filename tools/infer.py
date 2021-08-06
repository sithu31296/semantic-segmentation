import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import sys
sys.path.insert(0, '.')
from models import get_model
from utils.utils import time_sync
from datasets import __all__


class Model:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])
        # get dataset classes' colors
        self.palette = __all__[cfg['DATASET']['NAME']].PALETTE
        # initialize the model and load weights and send to device
        self.model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu')['state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        # preprocess parameters
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def resize(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        # resize the image
        image = T.Resize((nH, nW))(image)
        return image

    def preprocess(self, image: Tensor) -> Tensor:
        # resize the image to inference image size
        img = self.resize(image)
        # scale to [0.0, 1.0]
        img = img.float()
        img /= 255
        # normalize
        img = self.norm(img)
        # add batch size and send to device
        img = img.unsqueeze(0).to(self.device)
        return img

    def postprocess(self, seg_map: Tensor, orig_size: list) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_size, mode='bilinear', align_corners=False)
        # get segmentation map (value being 0 to num_classes)
        seg_map = F.softmax(seg_map, dim=1)
        seg_map = seg_map.argmax(dim=1).to(int)
        # convert segmentation map to color map
        seg_map = self.palette[seg_map]
        return seg_map.squeeze().cpu().permute(2, 0, 1).to(torch.uint8)
        
    @torch.no_grad()
    def predict(self, img_fname: str) -> Tensor:
        # read the image
        image = io.read_image(img_fname)
        # preprocess the image
        img = self.preprocess(image)
        # model pass
        start = time_sync()
        seg_map = self.model(img)
        print(f"Model Inference Time: {(time_sync()-start)*1000}ms.")
        # postprocess the image
        seg_map = self.postprocess(seg_map, image.shape[1:])
        return seg_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)
    
    model = Model(cfg)

    if cfg['TEST']['MODE'] == 'image':
        test_file = Path(cfg['TEST']['FILE'])
        if test_file.is_file():
            print(f'Inferencing {test_file} >>>', end=' ')
            segmap = model.predict(str(test_file))
            io.write_png(segmap, str(save_dir / f"{str(test_file.stem)}.png"))
        
        else:
            files = test_file.glob('*.*')
            for file in files:
                print(f'Inferencing {file} >>>', end=' ')
                segmap = model.predict(str(file))
                io.write_png(segmap, str(save_dir / f"{str(file.stem)}.png"))

    elif cfg['TEST']['MODE'] == 'video':
        pass
    else:
        pass

