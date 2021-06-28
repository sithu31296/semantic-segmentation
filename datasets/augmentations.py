import torchvision.transforms.functional as TF 
import random
import torch
from PIL import Image
from typing import Tuple, List, Union, Tuple, Optional


class Compose:
    def __init__(self, transforms: list) -> None:
        """
        Args:
            transforms: torchvision functional transforms in list
        """
        self.transforms = transforms

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert img.shape == mask.shape

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask


class adjustBrightness:
    def __init__(self, brightness_factor: float) -> None:
        """
        Args:
            brightness_factor: Can be non-negative number. 
                                0 gives a black image, 1 gives the original image
                                while 2 increases the brightness by a factor of 2.
        """
        self.brightness_factor = brightness_factor

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.adjust_brightness(img, self.brightness_factor), mask


class adjustContrast:
    def __init__(self, contrast_factor: float) -> None:
        """
        Args:
            contrast_factor: Can be non-negative number. 
                                0 gives a solid gray image, 1 gives the original image
                                while 2 increases the contrast by a factor of 2.
        """
        self.contrast_factor = contrast_factor

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.adjust_contrast(img, self.contrast_factor), mask


class adjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class adjustHue:
    def __init__(self, hue_factor: float) -> None:
        """
        The image hue is adjusted by converting the image to HSV and cyclically shifting the intensities in the hue channel(H).
        The image is then converted back to original image mode.

        Args:
            hue_factor: Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal hue channel in HSV space 
                        in positive and negative direction respectively. 0 means no shift. Therefore, 
                        both -0.5 and 0.5 will give an image with complementary colors while 0 gives the original image.
        """
        self.hue_factor = hue_factor

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.adjust_hue(img, self.hue_factor), mask


class adjustSaturation:
    def __init__(self, saturation_factor: float) -> None:
        """
        Args:
            saturation_factor: 0 gives a black and whie image, 1 gives the original image and 2 will enhance the saturation by a factor of 2.
        """
        self.saturation_factor = saturation_factor

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.adjust_saturation(img, self.saturation_factor), mask


class centerCrop:
    def __init__(self, output_size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.center_crop(img, self.output_size), TF.center_crop(mask, self.output_size)


class crop:
    def __init__(self, top: int, left: int, height: int, width: int) -> None:
        """Crops the given image at specified location and output size.

        Args:
            top: Vertical component of the top left corner of the crop box.
            left: Horizontal component of the top left corner of the crop box.
            height: Height of the crop box.
            width: Width of the crop box.
        """
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.crop(img, self.top, self.left, self.height, self.width), TF.crop(mask, self.top, self.left, self.height, self.width)


class randomCrop:
    def __init__(self, output_size: Union[int, List[int], Tuple[int]], p: float = 0.2) -> None:
        """Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert img.shape == mask.shape, "Image size and Label mask size should be equal"
        _, height, width = img.shape
        target_height, target_width = self.output_size

        if height < target_height or width < target_width:
            return img, mask
        else:
            if random.random() < self.p:
                new_top = random.randint(0, height - target_height)
                new_left = random.randint(0, width - target_width)
                return TF.crop(img, new_top, new_left, target_height, target_width), TF.crop(mask, new_top, new_left, target_height, target_width)
            else:
                return img, mask


class horizontalFlip:
    def __init__(self) -> None:
        pass

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.hflip(img), TF.hflip(mask)


class randomHorizontalFlip:
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (TF.hflip(img), TF.hflip(mask)) if random.random() < self.p else (img, mask)


class verticalFlip:
    def __init__(self) -> None:
        pass

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.vflip(img), TF.vflip(mask)


class randomVerticalFlip:
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (TF.vflip(img), TF.vflip(mask)) if random.random() < self.p else (img, mask)


class pad:
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

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.pad(img, self.padding, self.fill, self.padding_mode), TF.pad(mask, self.padding, self.fill, self.padding_mode)


class resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.resize(img, self.size, Image.BILINEAR), TF.resize(mask, self.size, Image.NEAREST)


class resizedCrop:
    def __init__(self, top: int, left: int, height: int, width: int, size: Union[int, Tuple[int], List[int]]) -> None:
        """Crop the given image and resize it to desired size.

        Args:
            top: Vertical component of the top left corner of the crop box.
            left: Horizontal component of the top left corner of the crop box.
            height: Height of the crop box.
            width: Width of the crop box.
            size: Desired output size. Same semantics as resize.
        """
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.size = size

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.resized_crop(img, self.top, self.left, self.height, self.width, self.size, Image.BILINEAR), TF.resized_crop(mask, self.top, self.left, self.height, self.width, self.size, Image.NEAREST)


class rotate:
    def __init__(self, angle: float, expand: bool = False, center: Optional[List[int]] = None) -> None:
        """Rotate the image by angle.

        Args:
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand falg assumes rotation around the center and no translation.
            center: Optional center of rotation. Origin is the upper left corner. Default is the center of the image.
        """
        self.angle = angle
        self.expand = expand
        self.center = center

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return TF.rotate(img, self.angle, Image.BILINEAR, self.expand, self.center), TF.rotate(mask, self.angle, Image.NEAREST, self.expand, self.center)


class randomRotate:
    def __init__(self, angle: float = 10, p: float = 0.2, expand: bool = False) -> None:
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
        self.angle = angle
        self.expand = expand

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        random_angle = random.random() * 2 * self.angle - self.angle
        return (TF.rotate(img, random_angle, Image.BILINEAR, self.expand), TF.rotate(mask, random_angle, Image.NEAREST, self.expand)) if random.random() < self.p else (img, mask)

