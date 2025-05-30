import numpy as np
from PIL import Image
class ImageUtils:
    def unpack_rgb_image(plane):
        try:
            """Return a correctly shaped numpy array given the image bytes."""
            image = Image.frombytes('RGB', (64, 64), plane.data)
            data = np.array(image)
        except Exception as e:
            print(f"unpack error {e}")
        return data

    # Given an s2clientprotocol.common_pb2.ImageData object, return a numpy array
    # The data is assumed to be an rgb image.


    def unpack_grayscale_image(plane):
        """Return a correctly shaped numpy array given the image bytes."""
        image = Image.frombytes('L', (64, 64), plane.data)
        data = np.array(image)
        return data