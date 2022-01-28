import argparse
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from semseg.utils.visualize import generate_palette
from semseg.utils.utils import timer


class Inference:
    def __init__(self, model: str) -> None:
        self.interpreter = tflite.Interpreter(model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.palette = generate_palette(self.output_details['shape'][-1], background=True)
        self.img_size = self.input_details['shape'][1:3]
        self.mean = np.array([0.485, 0.456, 0.406])[None, None, :]
        self.std = np.array([0.229, 0.224, 0.225])[None, None, :]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.img_size)
        image = np.array(image, dtype=np.float32)
        image /= 255
        image -= self.mean
        image /= self.std
        if self.input_details['dtype'] == np.int8 or self.input_details['dtype'] == np.uint8:
            scale, zero_point = self.input_details['quantization']
            image /= scale
            image += zero_point
            image = image.astype(self.input_details['dtype'])
        return image[np.newaxis, ...]

    def postprocess(self, seg_map: np.ndarray) -> np.ndarray:
        if self.output_details['dtype'] == np.int8 or self.output_details['dtype'] == np.uint8:
            scale, zero_point = self.output_details['quantization']
            seg_map = scale * (seg_map - zero_point)
        seg_map = np.argmax(seg_map, axis=-1).astype(int)
        seg_map = self.palette[seg_map]
        return seg_map.squeeze()

    @timer
    def model_forward(self, img: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details['index'])

    def predict(self, img_path: str) -> Image.Image:
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        seg_map = self.model_forward(image)
        seg_map = self.postprocess(seg_map)
        return seg_map.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='output/ddrnet_tflite2/ddrnet_float16.tflite')
    parser.add_argument('--img-path', type=str, default='assests/faces/27409477_1.jpg')
    args = parser.parse_args()

    session = Inference(args.model)
    seg_map = session.predict(args.img_path)
    seg_map = Image.fromarray(seg_map)
    seg_map.save(f"{args.img_path.split('.')[0]}_out.png")