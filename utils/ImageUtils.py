import numpy as np
from PIL import Image
class ImageUtils:
    def unpack_rgb_image(self, plane):
        try:
            """Return a correctly shaped numpy array given the image bytes."""
            image = Image.frombytes('RGB', (64, 64), plane.data)
            data = np.array(image)
        except Exception as e:
            print(f"unpack error {e}")
        return data

    def unpack_grayscale_image(self, plane):
        """Return a correctly shaped numpy array given the image bytes."""
        image = Image.frombytes('L', (64, 64), plane.data)
        data = np.array(image)
        return data