import torch
from PIL import ImageColor


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    # https://github.com/kornia/kornia/blob/e461f92ff9ee035d2de2513859bee4069356bc25/kornia/color/ycbcr.py
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta: float = 128.0
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3).floor()


def chroma_key(
    image: torch.Tensor,
    keycolor: list[str],
    tola: torch.Tensor,
    tolb: torch.Tensor,
    background_image: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized chroma key using pytorch, supports extra batch dimension.

    Args:
        image: Shape (B, C, H, W), dtype=torch.float32, range=[0, 1].
        keycolor: List for hex color codes.
        tola: Shape (B,)
        tolb: Shape (B,)
        background_image: Shape (B, C, H, W). Defaults to None.

    Returns:
        out: Shape (B, C, H, W).
        mask: Shape (B, H, W).
    """

    image = image * 255
    if background_image is not None:
        background_image = background_image * 255

    image_ycbcr = rgb_to_ycbcr(image)
    keycolor_rgb = torch.as_tensor(
        [ImageColor.getrgb(kc) for kc in keycolor], device=image.device
    )[..., None, None].float()
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
    mask = mask[:, None, :, :]

    out = torch.zeros_like(image, dtype=torch.float32)
    out = image - mask * keycolor_rgb
    if background_image is not None:
        out += mask * background_image
    out = out / 255
    out = torch.clip(out, 0, 1)

    return out, mask[:, 0, :, :]
