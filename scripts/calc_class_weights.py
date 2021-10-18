import torch
from torchvision import io


def calc_class_weights(files, n_classes):
    pixels = {}
    for file in files:
        lbl_path = str(file).split('.')[0].replace('images', 'labels')
        label = io.read_image(lbl_path)
        for i in range(n_classes):
            if pixels.get(i) is not None:
                pixels[i] += [(label == i).sum()]
            else:
                pixels[i] = [(label == i).sum()]
    
    class_freq = torch.tensor([sum(v).item() for v in pixels.values()])
    weights = 1 / torch.log1p(class_freq)
    weights *= n_classes 
    weights /= weights.sum()
    return weights