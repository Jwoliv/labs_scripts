import cv2
import random
from utils import text_to_bin

# Послідовне вбудовування повідомлення
def embed_sequential(image, message):
    bin_msg = text_to_bin(message)
    img_mod = image.copy()
    rows, cols, channels = img_mod.shape

    for i in range(len(bin_msg)):
        row, col = divmod(i, cols)
        if row >= rows:
            break
        pixel_value = img_mod[row, col].copy()  # Копіюємо значення пікселя
        pixel_value[0] = (pixel_value[0] & 254) | int(bin_msg[i])  # Модифікуємо лише синій канал
        img_mod[row, col] = pixel_value

    return img_mod

# Випадкове вбудовування повідомлення
def embed_random(image, message, seed):
    bin_msg = text_to_bin(message)
    img_mod = image.copy()
    rows, cols, channels = img_mod.shape
    total_pixels = rows * cols

    random.seed(seed)
    positions = random.sample(range(total_pixels), len(bin_msg))

    for i, pos in enumerate(positions):
        row, col = divmod(pos, cols)
        pixel_value = img_mod[row, col].copy()
        pixel_value[0] = (pixel_value[0] & 254) | int(bin_msg[i])  # Модифікуємо лише синій канал
        img_mod[row, col] = pixel_value

    return img_mod