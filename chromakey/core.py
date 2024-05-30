from PIL import Image, ImageColor
import numpy as np
from scipy.ndimage import gaussian_filter

def chroma_key(
    image: np.ndarray,
    keycolor: str = "#00FF00",
    background_image: np.ndarray = None,
    tola: int = 10,
    tolb: int = 30,
    gaussian_filter_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Chroma key.

    Args:
        image: Shape (H, W, C)
        keycolor: Hex color code. Defaults to "#00FF00".
        background_image: Shape (H, W, C). Defaults to None.
        tola: Tolerance 1. Defaults to 10.
        tolb: Tolerance 2. Defaults to 30.
        gaussian_filter_sigma: Gaussian blur sigma. Defaults to 1.5.

    Returns:
        out: Shape (H, W, C).
        mask: Shape (H, W).
    """
    image_pil = Image.fromarray(image)
    image_ycbcr = np.asarray(image_pil.convert("YCbCr"), dtype=np.float32)

    keycolor_pil = Image.new("RGB", (1, 1), keycolor)
    keycolor_rgb = np.asarray(keycolor_pil, dtype=np.float32).flatten()
    keycolor_ycbcr = np.asarray(
        keycolor_pil.convert("YCbCr"), dtype=np.float32
    ).flatten()

    dist = np.sqrt(
        (image_ycbcr[:, :, 1] - keycolor_ycbcr[1]) ** 2
        + (image_ycbcr[:, :, 2] - keycolor_ycbcr[2]) ** 2
    )

    mask = np.zeros_like(dist, dtype=np.float32)
    mask[dist > tola] = 0
    mask[(dist >= tola) & (dist < tolb)] = (
        dist[(dist >= tola) & (dist < tolb)] - tola
    ) / (tolb - tola)
    mask[dist >= tolb] = 1
    mask = 1 - mask
    mask = gaussian_filter(mask, sigma=gaussian_filter_sigma)

    out = np.zeros_like(image, dtype=np.float32)
    out = image - mask[..., None] * keycolor_rgb
    if background_image is not None:
        out += mask[..., None] * background_image
    out = np.clip(out, 0, 255)
    out = np.uint8(out)

    return out, mask

def chroma_key_vectorized(
    image: "torch.Tensor",
    keycolor: list[str],
    tola: "torch.Tensor",
    tolb: "torch.Tensor",
    background_image: "torch.Tensor" = None,
    gaussian_filter_kernel_size: tuple[int] = (13, 13),
    gaussian_filter_sigma: float = (1.5, 1.5),
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Vectorized chroma key using pytorch and kornia, supports extra batch dimension.

    Args:
        image: Shape (B, C, H, W)
        keycolor: List for hex color codes.
        tola: Shape (B,)
        tolb: Shape (B,)
        background_image: Shape (B, C, H, W). Defaults to None.
        gaussian_filter_kernel_size: Gaussian blur kernel size. Defaults to (13, 13).
        gaussian_filter_sigma: Gaussian blur sigma. Defaults to (1.5, 1.5).

    Returns:
        out: Shape (B, C, H, W).
        mask: Shape (B, H, W).
    """
    import torch
    from torch import Tensor, nn

    # https://github.com/kornia/kornia/blob/e461f92ff9ee035d2de2513859bee4069356bc25/kornia/color/ycbcr.py
    def rgb_to_ycbcr(image: Tensor) -> Tensor:
        if not isinstance(image, Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        r: Tensor = image[..., 0, :, :]
        g: Tensor = image[..., 1, :, :]
        b: Tensor = image[..., 2, :, :]

        delta: float = 0.5
        y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
        cb: Tensor = (b - y) * 0.564 + delta
        cr: Tensor = (r - y) * 0.713 + delta
        return torch.stack([y, cb, cr], -3)

    image_ycbcr = rgb_to_ycbcr(image)
    keycolor_rgb = torch.as_tensor([ImageColor.getrgb(kc) for kc in keycolor], device=image.device)[..., None, None]
    keycolor_ycbcr = rgb_to_ycbcr(keycolor_rgb)

    dist = torch.sqrt(
        (image_ycbcr[:, 1, :, :] - keycolor_ycbcr[:, 1]) ** 2
        + (image_ycbcr[:, 2, :, :] - keycolor_ycbcr[:, 2]) ** 2
    )

    mask = torch.zeros_like(dist, dtype=torch.float32)
    mask[dist > tola] = 0
    mask[(dist >= tola) & (dist < tolb)] = (
        dist[(dist >= tola) & (dist < tolb)] - tola
    ) / (tolb - tola)
    mask[dist >= tolb] = 1
    mask = 1 - mask
    # mask = gaussian_blur2d(mask[:, None, :, :], gaussian_filter_kernel_size, gaussian_filter_sigma)
    mask = mask[:, None, :, :]

    out = torch.zeros_like(image, dtype=torch.float32)
    out = image - mask * keycolor_rgb
    if background_image is not None:
        out += mask * background_image
    out = torch.clip(out, 0, 255)
    out = out.to(torch.uint8)

    return out, mask[:, 0, :, :]
