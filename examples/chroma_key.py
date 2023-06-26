from PIL import Image
from pathlib import Path
import numpy as np
from chromakey import chroma_key

# Image from
# https://sotamedialab.wordpress.com/2017/02/14/common-green-screen-photography-mistakes/
image_path = Path(__file__).parent / "../uneven_green_green_lighting.png"
image = Image.open(str(image_path))
background_image = Image.new("RGB", image.size, "red")

image = np.asarray(image)
background_image = np.asarray(background_image)
masked_image, _ = chroma_key(
    image, keycolor="#6dff8b", background_image=background_image, tola=25, tolb=50
)
Image.fromarray(masked_image).show()
