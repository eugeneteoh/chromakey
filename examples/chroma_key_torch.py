from pathlib import Path

import numpy as np
import torch
from PIL import Image

from chromakey.torch import chroma_key

# Image from
# https://sotamedialab.wordpress.com/2017/02/14/common-green-screen-photography-mistakes/
image_path = Path(__file__).parent / "../uneven_green_green_lighting.png"
image = Image.open(str(image_path))

background_image = Image.new("RGB", image.size, "red")

image = torch.tensor(np.asarray(image)).unsqueeze(0)
image = torch.einsum("bhwc->bchw", image)
image = image.float() / 255
background_image = torch.tensor(np.asarray(background_image)).unsqueeze(0)
background_image = torch.einsum("bhwc->bchw", background_image)
background_image = background_image.float() / 255
masked_image, _ = chroma_key(
    image,
    keycolor=["#6dff8b"],
    background_image=background_image,
    tola=torch.as_tensor([25]),
    tolb=torch.as_tensor([50]),
)
masked_image = (masked_image * 255).to(torch.uint8)
print(masked_image.shape)
Image.fromarray(masked_image.numpy(force=True)[0].transpose(1, 2, 0)).show()
