import cv2
import numpy as np
from matplotlib import pyplot as plt


# Читання JPG зображення
def read_jpg(filename):
    return cv2.imread(filename)

# Запис JPG зображення
def write_jpg(filename, image, quality=95):
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

# Перетворення тексту в бінарний формат
def text_to_bin(text):
    return ''.join(format(ord(c), '08b') for c in text)

# Перетворення бінарного формату в текст
def bin_to_text(binary):
    chars = [binary[i:i + 8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

# Обчислення MSE (середньоквадратичної помилки) між двома зображеннями
def mse(image1, image2):
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)

# Аналіз гістрограми
def plot_histogram(original, modified, title="Histogram Comparison"):
    plt.figure(figsize=(10, 5))

    # Аналізуємо лише синій канал (індекс 0 у BGR)
    plt.hist(original[:, :, 0].ravel(), bins=256, alpha=0.7, label='Original', color='blue', edgecolor='black',
             linewidth=0.5)
    plt.hist(modified[:, :, 0].ravel(), bins=256, alpha=0.7, label='Modified', color='orange', edgecolor='black',
             linewidth=0.5)

    plt.title(title)
    plt.xlabel('Pixel Intensity (Blue Channel)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()