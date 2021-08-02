import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if self.brightness > 0:
            img = TF.adjust_brightness(img, self.brightness)
        if self.contrast > 0:
            img = TF.adjust_contrast(img, self.contrast)
        if self.saturation > 0:
            img = TF.adjust_saturation(img, self.saturation)
        if self.hue > 0:
            img = TF.adjust_hue(img, self.hue)
        return img, mask


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.adjust_sharpness(img, self.sharpness)
        return img, mask


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.autocontrast(img)
        return img, mask


class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)


class RandomGaussianBlur:
    def __init__(self, kernel_size: List[int], sigma: Optional[List[float]] = None, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.gaussian_blur(img, self.kernel_size, self.sigma)
        return img, mask


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.2) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        _, height, width = img.shape
        target_height, target_width = self.size

        if height > target_height or width > target_width:
            if random.random() < self.p:
                new_top = random.randint(0, height - target_height)
                new_left = random.randint(0, width - target_width)
                return TF.crop(img, new_top, new_left, target_height, target_width), TF.crop(mask, new_top, new_left, target_height, target_width)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class Normalize:
    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.normalize(img, self.mean, self.std), mask


class Pad:
    def __init__(self, padding: Union[List[int], Tuple[int], int], fill: int = 0, padding_mode: str = 'constant') -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            padding: Padding on each border. 
                    If a single int is provided, this is used to pad all borders.
                    If a tuple of length 2, this is the padding on left/right and top/bottom respectively.
                    If a tuple of length 4, this is the padding for left, top, right, and bottom respectively.
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
            padding_mode: Type of padding. Should be constant, edge, or reflect. Default is constant. 
                constant: pads with a constant value, this value is specified with fill
                edge: pads with the last value on the edge of the image
                reflect: pads with reflection of image (without repeating the last value on the dege)
        """
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.pad(img, self.padding, self.fill, self.padding_mode), TF.pad(mask, self.padding, self.fill, self.padding_mode)


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.resize(img, self.size, TF.InterpolationMode.BILINEAR), TF.resize(mask, self.size, TF.InterpolationMode.NEAREST)


class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = img.float()
        img /= 255
        img = TF.normalize(img, self.mean, self.std)
        return img, mask
    

class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        width, height = TF._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return TF.resized_crop(img, i, j, h, w, self.size, TF.InterpolationMode.BILINEAR), TF.resized_crop(mask, i, j, h, w, self.size, TF.InterpolationMode.NEAREST)


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        random_angle = random.random() * 2 * self.angle - self.angle
        return (TF.rotate(img, random_angle, TF.InterpolationMode.BILINEAR, self.expand), TF.rotate(mask, random_angle, TF.InterpolationMode.NEAREST, self.expand)) if random.random() < self.p else (img, mask)

