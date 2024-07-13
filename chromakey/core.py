from PIL import Image
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

