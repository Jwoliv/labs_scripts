from datetime import datetime

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


# Читання JPG зображення
def read_jpg(filename):
    # Читає зображення в режимі відтінків сірого
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


# Запис JPG зображення
def write_jpg(filename, image, quality=95):
    # Записує зображення у файл із заданою якістю
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


# Перетворення тексту в бінарний формат
def text_to_bin(text):
    # Перетворює рядок тексту в бінарну строку
    return ''.join(format(ord(c), '08b') for c in text)


# Перетворення бінарного формату в текст
def bin_to_text(binary):
    # Перетворює бінарну строку назад у текст, розбиваючи на 8-бітні частини
    chars = [binary[i:i + 8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)


# Послідовне вбудовування LSB (Least Significant Bit)
def embed_sequential(image, message):
    # Вбудовує повідомлення в зображення за допомогою послідовного підходу до LSB
    bin_msg = text_to_bin(message)  # Перетворюємо повідомлення в бінарний формат
    img_mod = image.copy()  # Робимо копію зображення для модифікацій
    rows, cols = img_mod.shape  # Отримуємо розміри зображення

    # Проходимо кожен біт повідомлення і вбудовуємо його в зображення
    for i in range(len(bin_msg)):
        row, col = divmod(i, cols)  # Рахуємо індекси рядка і стовпця на основі позиції в повідомленні
        if row >= rows:  # Перевірка, щоб не вийти за межі зображення
            break
        # Модифікуємо LSB пікселя, щоб вбудувати біт повідомлення
        img_mod[row, col] = (img_mod[row, col] & 254) | int(bin_msg[i])

    return img_mod


# Випадкове вбудовування LSB
def embed_random(image, message, seed):
    # Вбудовує повідомлення в зображення за допомогою випадкового підходу до LSB
    bin_msg = text_to_bin(message)  # Перетворюємо повідомлення в бінарний формат
    img_mod = image.copy()  # Робимо копію зображення для модифікацій
    rows, cols = img_mod.shape  # Отримуємо розміри зображення
    total_pixels = rows * cols  # Загальна кількість пікселів у зображенні

    # Встановлюємо сид для генератора випадкових чисел для відтворюваності результату
    random.seed(seed)
    # Випадковим чином вибираємо позиції пікселів для вбудовування повідомлення
    positions = random.sample(range(total_pixels), len(bin_msg))

    # Вбудовуємо повідомлення в випадкові пікселі
    for i, pos in enumerate(positions):
        row, col = divmod(pos, cols)  # Рахуємо індекси рядка і стовпця за випадковою позицією
        # Модифікуємо LSB пікселя, щоб вбудувати біт повідомлення
        img_mod[row, col] = (img_mod[row, col] & 254) | int(bin_msg[i])

    return img_mod


# Витягування прихованого повідомлення
def extract_message(image, length):
    # Витягує приховане повідомлення з зображення, використовуючи послідовне витягування LSB
    bin_msg = ''
    rows, cols = image.shape  # Отримуємо розміри зображення

    # Проходимо кожен піксель і витягуємо LSB (найменш значущий біт)
    for i in range(length * 8):
        row, col = divmod(i, cols)  # Рахуємо індекси рядка і стовпця
        if row >= rows:  # Перевірка, щоб не вийти за межі зображення
            break
        # Витягуємо LSB з пікселя
        bin_msg += str(image[row, col] & 1)

    return bin_to_text(bin_msg)  # Перетворюємо бінарну строку назад у текст


# Витягування прихованого повідомлення з випадково вбудованого зображення
def extract_message_random(image, length, seed):
    # Витягує приховане повідомлення з зображення з випадковим вбудовуванням LSB
    bin_msg = ''
    rows, cols = image.shape  # Отримуємо розміри зображення
    total_pixels = rows * cols  # Загальна кількість пікселів

    # Встановлюємо сид для генератора випадкових чисел
    random.seed(seed)
    # Випадковим чином вибираємо пікселі для витягування повідомлення
    positions = random.sample(range(total_pixels), length * 8)

    # Витягуємо LSB з вибраних випадкових пікселів
    for pos in positions:
        row, col = divmod(pos, cols)  # Рахуємо індекси рядка і стовпця за позицією
        bin_msg += str(image[row, col] & 1)

    return bin_to_text(bin_msg)  # Перетворюємо бінарну строку назад у текст


# Обчислення MSE (середньоквадратичної помилки) між двома зображеннями
def mse(image1, image2):
    # Обчислює середньоквадратичну помилку між двома зображеннями
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)


# Аналіз гістрограми
def plot_histogram(original, modified):
    # Побудова гістрограм для оригінального і зміненого зображення для порівняння
    plt.figure(figsize=(10, 5))
    plt.hist(original.ravel(), bins=256, alpha=0.5, label='Original')  # Гістрограма оригінального зображення
    plt.hist(modified.ravel(), bins=256, alpha=0.5, label='Modified')  # Гістрограма зміненого зображення
    plt.legend()  # Додаємо легенду
    plt.show()


# Основне виконання
if __name__ == '__main__':
    image = read_jpg("images/image_2.jpg")  # Читаємо зображення з файлу
    message = """
        Lorem Ipsum is simply dummy text of the printing and typesetting industry.
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
        when an unknown printer took a galley of type and scrambled it to make a type specimen book.
        It has survived not only five centuries, but also the leap into electronic typesetting,
        remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset
        sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like
        Aldus PageMaker including versions of Lorem Ipsum.
        """  # Повідомлення, яке будемо вбудовувати

    # Послідовне вбудовування: Вбудовуємо повідомлення послідовно в зображення
    image_seq = embed_sequential(image, message)
    write_jpg(f"./images/image_seq_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg", image_seq)

    # Випадкове вбудовування: Вбудовуємо повідомлення випадковим чином в зображення, використовуючи ключ
    key = 340698234968  # Ключ, використаний для випадкового вбудовування
    image_rand = embed_random(image, message, key)
    write_jpg(f"./images/image_random_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg", image_rand)

    # Витягування повідомлень
    retrieved_msg_seq = extract_message(image_seq, len(message))  # Витягуємо повідомлення з послідовно вбудованого зображення
    print("Retrieved Message (Sequential):", retrieved_msg_seq)

    retrieved_msg_rand = extract_message_random(image_rand, len(message), key)  # Витягуємо повідомлення з випадково вбудованого зображення
    print("Retrieved Message (Random):", retrieved_msg_rand)

    # Аналіз змін зображень через гістрограми
    plot_histogram(image, image_seq)  # Порівнюємо гістрограми оригінального і послідовно вбудованого зображення
    print("MSE (Sequential):", mse(image, image_seq))  # Обчислюємо MSE між оригінальним і послідовно вбудованим зображенням
    print("MSE (Random):", mse(image, image_rand))  # Обчислюємо MSE між оригінальним і випадково вбудованим зображенням
