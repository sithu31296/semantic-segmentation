import os
import torch
import argparse
import yaml

from PIL import Image
from torchvision import io
from torchvision import transforms

from models import choose_models
from datasets import choose_datasets


class Model:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['TEST']['DEVICE'])

        _, self.colors = choose_datasets[cfg['DATASET']['NAME']].colorMap()

        self.model = choose_models[cfg['MODEL']['NAME']](cfg['DATASET']['NUM_CLASSES'])
        self.model.load_state_dict(torch.load(cfg['TEST']['TRAINED_MODEL'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_transforms = transforms.Compose([
            transforms.Resize(cfg['TEST']['IMG_SIZE'], interpolation=Image.BILINEAR),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def decode_segmap(self, label):
        return self.colors[label.to(int)]
        
    @torch.no_grad()
    def predict(self, image):
        image = image.float()
        image /= 255
        image = self.image_transforms(image).unsqueeze(0).to(self.device)
        pred = self.model(image)
        pred = self.decode_segmap(pred).cpu().squeeze().permute(2, 0, 1).type(torch.uint8)
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/camvid.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(cfg['TEST']['OUTPUT']): os.makedirs(cfg['TEST']['OUTPUT'])
    
    model = Model(cfg)

    if cfg['TEST']['MODE'] == 'image':
        
        if os.path.isfile(cfg['TEST']['FILE']):
            image = io.read_image(cfg['TEST']['FILE'])
            segmap = model.predict(image)
            io.write_png(segmap, os.path.join(cfg['TEST']['OUTPUT'], os.path.basename(cfg['TEST']['FILE']).rsplit('.', maxsplit=1)[0]+'.png'))
        
        else:
            files = os.listdir(cfg['TEST']['FILE'])
            for file in files:
                image = io.read_image(os.path.join(cfg['TEST']['FILE'], file))
                segmap = model.predict(image)
                io.write_png(segmap, os.path.join(cfg['TEST']['OUTPUT'], os.path.basename(file).rsplit('.', maxsplit=1)[0]+'.png'))

    elif cfg['TEST']['MODE'] == 'video':
        pass
    else:
        pass

