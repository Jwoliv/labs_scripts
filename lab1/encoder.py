import cv2
import random
from utils import text_to_bin

# Послідовне вбудовування повідомлення
def embed_sequential(image, message):
    bin_msg = text_to_bin(message)  # Перетворюємо повідомлення в бінарний формат
    img_mod = image.copy()  # Робимо копію зображення для модифікацій
    rows, cols, channels = img_mod.shape  # Отримуємо розміри зображення та кількість каналів

    for i in range(len(bin_msg)):
        row, col = divmod(i, cols)  # Рахуємо індекси рядка і стовпця
        if row >= rows:  # Перевірка, щоб не вийти за межі зображення
            break
        pixel_value = img_mod[row, col]
        img_mod[row, col] = (pixel_value & 254) | int(bin_msg[i])  # Оновлюємо LSB кожного каналу

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
        pixel_value = img_mod[row, col]
        img_mod[row, col] = (pixel_value & 254) | int(bin_msg[i])

    return img_mod
