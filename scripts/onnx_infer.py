import argparse
import numpy as np
import onnxruntime
from PIL import Image
from semseg.utils.visualize import generate_palette
from semseg.utils.utils import timer


class Inference:
    def __init__(self, model: str) -> None:
        self.session = onnxruntime.InferenceSession(model)
        self.input_details = self.session.get_inputs()[0]
        self.palette = generate_palette(self.session.get_outputs()[0].shape[1], background=True)
        self.img_size = self.input_details.shape[-2:]
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
        return self.session.run(None, {self.input_details.name: img})[0]

    def predict(self, img_path: str) -> Image.Image:
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        seg_map = self.model_forward(image)
        seg_map = self.postprocess(seg_map)
        return seg_map.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='output/DDRNet_23slim_HELEN_59_75.onnx')
    parser.add_argument('--img-path', type=str, default='assests/faces/27409477_1.jpg')
    args = parser.parse_args()

    session = Inference(args.model)
    seg_map = session.predict(args.img_path)
    seg_map = Image.fromarray(seg_map)
    seg_map.save(f"{args.img_path.split('.')[0]}_out.png")