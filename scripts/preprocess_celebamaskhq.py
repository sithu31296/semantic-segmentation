import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image


def main(root):
    root = Path(root)
    annot_dir = root / 'CelebAMask-HQ-label'
    annot_dir.mkdir(exist_ok=True)

    train_lists = []
    test_lists = []
    val_lists = []

    names = [
        'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
        'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth'
    ]
    num_images = 30000

    for folder in root.iterdir():
        if folder.is_dir():
            if folder.name == 'CelebAMask-HQ-mask-anno':
                print("Transforming separate masks into one-hot mask...")
                for i in tqdm(range(num_images)):
                    folder_num = i // 2000
                    label = np.zeros((512, 512))
                    for idx, name in enumerate(names):
                        fname = folder / f"{folder_num}" / f"{str(i).rjust(5, '0')}_{name}.png"
                        if fname.exists():
                            img = Image.open(fname).convert('P')
                            img = np.array(img)
                            label[img != 0] = idx + 1

                    label = Image.fromarray(label.astype(np.uint8))
                    label.save(annot_dir / f"{i}.png") 

    print("Splitting into train/val/test...")

    with open(root / "CelebA-HQ-to-CelebA-mapping.txt") as f:
        lines = f.read().splitlines()[1:]
        image_list = [int(line.split()[1]) for line in lines]
            

    for idx, fname in enumerate(image_list):
        if fname >= 162771 and fname < 182638:
            val_lists.append(f"{idx}\n")

        elif fname >= 182638:
            test_lists.append(f"{idx}\n")

        else:
            train_lists.append(f"{idx}\n")

    print(f"Train Size: {len(train_lists)}")
    print(f"Val Size: {len(val_lists)}")
    print(f"Test Size: {len(test_lists)}")

    with open(root / 'train_list.txt', 'w') as f:
        f.writelines(train_lists)

    with open(root / 'val_list.txt', 'w') as f:
        f.writelines(val_lists)

    with open(root / 'test_list.txt', 'w') as f:
        f.writelines(test_lists)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/sithu/datasets/CelebAMask-HQ')
    args = parser.parse_args()
    main(args.root)