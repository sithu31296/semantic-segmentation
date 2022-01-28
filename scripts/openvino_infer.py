import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from openvino.inference_engine import IECore
from semseg.utils.visualize import generate_palette
from semseg.utils.utils import timer


class Inference:
    def __init__(self, model: str) -> None:
        files = Path(model).iterdir()

        for file in files:
            if file.suffix == '.xml':
                model = str(file)
            elif file.suffix == '.bin':
                weights = str(file)
        ie = IECore()
        model = ie.read_network(model=model, weights=weights)
        self.input_info = next(iter(model.input_info))
        self.output_info = next(iter(model.outputs))
        self.img_size = model.input_info['input'].input_data.shape[-2:]
        self.palette = generate_palette(11, background=True)
        self.engine = ie.load_network(network=model, device_name='CPU')
        
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.img_size)
        image = np.array(image, dtype=np.float32).transpose(2, 0, 1)
        image /= 255
        image -= self.mean
        image /= self.std
        image = image[np.newaxis, ...]
        return image

    def postprocess(self, seg_map: np.ndarray) -> np.ndarray:
        seg_map = np.argmax(seg_map, axis=1).astype(int)
        seg_map = self.palette[seg_map]
        return seg_map.squeeze()

    @timer
    def model_forward(self, img: np.ndarray) -> np.ndarray:
        return self.engine.infer(inputs={self.input_info: img})[self.output_info]

    def predict(self, img_path: str) -> Image.Image:
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        seg_map = self.model_forward(image)
        seg_map = self.postprocess(seg_map)
        return seg_map.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='output/ddrnet_openvino')
    parser.add_argument('--img-path', type=str, default='assests/faces/27409477_1.jpg')
    args = parser.parse_args()

    session = Inference(args.model)
    seg_map = session.predict(args.img_path)
    seg_map = Image.fromarray(seg_map)
    seg_map.save(f"{args.img_path.split('.')[0]}_out.png")
