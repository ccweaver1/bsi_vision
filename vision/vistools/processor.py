"""
Process an image that we can pass to our networks.
"""
# docker install PIL (workaround cause requirements file seems to not be working)

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.call(['pip', 'install', 'Pillow'])
    from PIL import Image
    
import numpy as np

def process_image(im, target_shape):
    """Given an image filename, process it and return the array."""
    h, w = target_shape
    # Load the image.
    if isinstance(im, np.ndarray):
        image = Image.fromarray(im)
    else:
        image = Image.open(im)
    
    image = image.resize((w,h))

    # Turn it into numpy, normalize and return.
    img_arr = np.asarray(image, dtype=np.float32)
    x = (img_arr / 255.).astype(np.float32)

    return x
