import numpy as np
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.state import State
import numbers
import numba as nb
from numba import njit

@njit(parallel=False, fastmath=True, inline="always")
def apply_cj(
    im,
    apply_bri,
    bri_ratio,
    apply_cont,
    cont_ratio,
    apply_sat,
    sat_ratio,
    apply_hue,
    hue_factor,
):

    gray = (
        np.float32(0.2989) * im[..., 0]
        + np.float32(0.5870) * im[..., 1]
        + np.float32(0.1140) * im[..., 2]
    )
    one = np.float32(1)
    # Brightness
    if apply_bri:
        im = im * bri_ratio

    # Contrast
    if apply_cont:
        im = cont_ratio * im + (one - cont_ratio) * np.float32(gray.mean())

    # Saturation
    if apply_sat:
        im[..., 0] = sat_ratio * im[..., 0] + (one - sat_ratio) * gray
        im[..., 1] = sat_ratio * im[..., 1] + (one - sat_ratio) * gray
        im[..., 2] = sat_ratio * im[..., 2] + (one - sat_ratio) * gray

    # Hue
    if apply_hue:
        hue_factor_radians = hue_factor * 2.0 * np.pi
        cosA = np.cos(hue_factor_radians)
        sinA = np.sin(hue_factor_radians)
        v1, v2, v3 = 1.0 / 3.0, np.sqrt(1.0 / 3.0), (1.0 - cosA)
        hue_matrix = [
            [
                cosA + v3 / 3.0,
                v1 * v3 - v2 * sinA,
                v1 * v3 + v2 * sinA,
            ],
            [
                v1 * v3 + v2 * sinA,
                cosA + v1 * v3,
                v1 * v3 - v2 * sinA,
            ],
            [
                v1 * v3 - v2 * sinA,
                v1 * v3 + v2 * sinA,
                cosA + v1 * v3,
            ],
        ]
        hue_matrix = np.array(hue_matrix, dtype=np.float64).T
        for row in nb.prange(im.shape[0]):
            im[row] = im[row] @ hue_matrix
    return np.clip(im, 0, 255).astype(np.uint8)


class RandomColorJitter(Operation):
    """Add ColorJitter with probability jitter_prob.
    Operates on raw arrays (not tensors).

    see https://github.com/pytorch/vision/blob/28557e0cfe9113a5285330542264f03e4ba74535/torchvision/transforms/functional_tensor.py#L165
     and https://sanje2v.wordpress.com/2021/01/11/accelerating-data-transforms/
    Parameters
    ----------
    jitter_prob : float, The probability with which to apply ColorJitter.
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        jitter_prob=0.5,
        brightness=0.8,
        contrast=0.4,
        saturation=0.4,
        hue=0.2
    ):
        super().__init__()
        self.jitter_prob = jitter_prob

        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5))
        assert self.jitter_prob >= 0 and self.jitter_prob <= 1

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative."
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            setattr(self, f"apply_{name}", False)
        else:
            setattr(self, f"apply_{name}", True)
        return tuple(value)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()

        jitter_prob = self.jitter_prob

        apply_bri = self.apply_brightness
        bri = self.brightness

        apply_cont = self.apply_contrast
        cont = self.contrast

        apply_sat = self.apply_saturation
        sat = self.saturation

        apply_hue = self.apply_hue
        hue = self.hue

        def color_jitter(images, _):
            for i in my_range(images.shape[0]):
                if np.random.rand() > jitter_prob:
                    continue

                images[i] = apply_cj(
                    images[i].astype("float64"),
                    apply_bri,
                    np.random.uniform(bri[0], bri[1]),
                    apply_cont,
                    np.random.uniform(cont[0], cont[1]),
                    apply_sat,
                    np.random.uniform(sat[0], sat[1]),
                    apply_hue,
                    np.random.uniform(hue[0], hue[1]),
                )
            return images

        color_jitter.is_parallel = True
        return color_jitter

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)

class RandomGrayscale(Operation):
    '''
    Randomly convert image to grayscale with a probability of p (not tensors).

    Parameters
    ----------
    p : float
        probability to apply contrast
    '''
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def generate_code(self):
        my_range = Compiler.get_iterator()
        p = self.p

        def rgb_to_grayscale(images, *_):

            apply_grayscale = np.random.rand(images.shape[0]) < p
            for i in my_range(images.shape[0]):
                if apply_grayscale[i]:
                    r, g, b = images[i,:,:,0], images[i,:,:,1], images[i,:,:,2]
                    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).astype(images[i].dtype)
                    l_img = np.expand_dims(l_img, axis=-1)
                    l_img = np.broadcast_to(l_img, images[0].shape)
                    images[i] = l_img

            return images

        rgb_to_grayscale.is_parallel = True
        return rgb_to_grayscale

    def declare_state_and_memory(self, previous_state):
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))