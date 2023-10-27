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
    from kornia.color import rgb_to_ycbcr
    from kornia.filters import gaussian_blur2d

    image_ycbcr = rgb_to_ycbcr(image)
    keycolor_rgb = torch.as_tensor([ImageColor.getrgb(kc) for kc in keycolor])[..., None, None]
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
    mask = gaussian_blur2d(mask[:, None, :, :], gaussian_filter_kernel_size, gaussian_filter_sigma)

    out = torch.zeros_like(image, dtype=torch.float32)
    out = image - mask * keycolor_rgb
    if background_image is not None:
        out += mask * background_image
    out = torch.clip(out, 0, 255)
    out = out.to(torch.uint8)

    return out, mask[:, 0, :, :]