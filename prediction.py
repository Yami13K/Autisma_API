import base64

from PIL import Image
from io import BytesIO


def ReadImage(image_encoded):
    name = 'decoded_image.png'
    with open(name, 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(image_encoded)
        file_to_save.write(decoded_image_data)
    image = Image.open(name)
    return image
