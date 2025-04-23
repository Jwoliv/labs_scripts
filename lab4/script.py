import numpy as np
import cv2
from scipy.fftpack import dct, idct
import os


def resize_to_multiple_of_8(img_path, output_path):
    """Resize image to dimensions divisible by 8."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load {img_path}")

    height, width = image.shape
    new_height = ((height + 7) // 8) * 8
    new_width = ((width + 7) // 8) * 8

    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_img)
    print(f"Resized image to {new_width}x{new_height} and saved to {output_path}")
    return output_path


def convert_jpg_to_png(jpg_path, png_path):
    """Convert a JPG image to PNG format."""
    if not os.path.exists(jpg_path):
        raise FileNotFoundError(f"Input image {jpg_path} does not exist.")
    image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image {jpg_path}.")
    cv2.imwrite(png_path, image)
    return png_path


def text_to_bits(text):
    """Convert a text string to a binary string."""
    return ''.join(format(ord(c), '08b') for c in text)


def embed_koch_zhao(img_path, bits, output_path, P=100):
    """Embed a binary string into an image using the Koch-Zhao algorithm."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Input image {img_path} does not exist.")

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image {img_path}.")

    height, width = image.shape
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"Image dimensions {width}x{height} must be divisible by 8.")

    if not all(bit in '01' for bit in bits):
        raise ValueError("Message must be a binary string (0s and 1s).")
    if len(bits) > (height * width // 64):
        raise ValueError("Message is too long for the image capacity.")

    stego = np.copy(image).astype(np.float32)

    def embed_bit(_dct_block, bit, u1, v1, u2, v2):
        c1, c2 = _dct_block[u1, v1], _dct_block[u2, v2]
        if bit == '0':
            _dct_block[u1, v1] = c1 + P / 2
            _dct_block[u2, v2] = c2 - P / 2
        else:
            _dct_block[u1, v1] = c1 - P / 2
            _dct_block[u2, v2] = c2 + P / 2
        return _dct_block

    bit_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if bit_idx >= len(bits):
                break
            block = stego[i:i + 8, j:j + 8]
            if block.shape != (8, 8):
                continue
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_block = embed_bit(dct_block, bits[bit_idx], 2, 3, 3, 2)
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            stego[i:i + 8, j:j + 8] = np.clip(idct_block, 0, 255)
            bit_idx += 1

    cv2.imwrite(output_path, stego.astype(np.uint8))
    print(f"Stego image saved to {output_path}")


def extract_koch_zhao(stego_path, bit_length, P=100):
    """Extract a binary string from a stego image using the Koch-Zhao algorithm."""
    if not os.path.exists(stego_path):
        raise FileNotFoundError(f"Stego image {stego_path} does not exist.")

    image = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load stego image {stego_path}.")

    h, w = image.shape
    if bit_length > (h * w // 64):
        raise ValueError("Requested bit length exceeds image capacity.")

    bits = ''
    bit_count = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if bit_count >= bit_length:
                break
            block = image[i:i + 8, j:j + 8].astype(np.float32)
            if block.shape != (8, 8):
                continue
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            c1, c2 = dct_block[2, 3], dct_block[3, 2]
            bit = '0' if c1 > c2 else '1'
            bits += bit
            bit_count += 1

    return bits


try:
    os.makedirs("images", exist_ok=True)

    image_alias = 'image_2'

    source_jpg = f"images/{image_alias}.jpg"
    resized_jpg = f"images/{image_alias}_resized.jpg"
    converted_png = f"images/{image_alias}_converted.png"
    output_stego = f"images/{image_alias}_stego_koch.png"

    img = cv2.imread(source_jpg, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {source_jpg}")
    h, w = img.shape
    if h % 8 != 0 or w % 8 != 0:
        source_jpg = resize_to_multiple_of_8(source_jpg, resized_jpg)

    convert_jpg_to_png(source_jpg, converted_png)
    print(f"Converted {source_jpg} to {converted_png}")

    message_text = "lishchuk"
    message_bits = text_to_bits(message_text)
    print(f"Input text: {message_text}")
    print(f"Message bits: {message_bits}")

    embed_koch_zhao(converted_png, message_bits, output_stego)

    extracted_bits = extract_koch_zhao(output_stego, bit_length=len(message_bits))
    print("Embedded bits:", message_bits)
    print("Extracted bits:", extracted_bits)

    chars = [chr(int(extracted_bits[i:i+8], 2)) for i in range(0, len(extracted_bits), 8)]
    print("Extracted message as text:", ''.join(chars))

except Exception as e:
    print(f"Error: {str(e)}")