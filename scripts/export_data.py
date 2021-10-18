import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def create_calibrate_data(image_folder, save_path):
    dataset = []
    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std = np.array([0.229, 0.224, 0.225])[None, None, :]
    files = list(Path(image_folder).glob('*.jpg'))[:100]
    for file in tqdm(files):
        image = Image.open(file).convert('RGB')
        image = image.resize((512, 512))
        image = np.array(image, dtype=np.float32)
        image /= 255
        image -= mean
        image /= std
        dataset.append(image)
    dataset = np.stack(dataset, axis=0)
    np.save(save_path, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/home/sithu/datasets/SmithCVPR2013_dataset_resized/images')
    parser.add_argument('--save-path', type=str, default='output/calibrate_data')
    args = parser.parse_args()

    create_calibrate_data(args.dataset_path, args.save_path)

