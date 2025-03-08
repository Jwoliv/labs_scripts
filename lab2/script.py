import os
import time
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

UINT_8 = 'uint8' # кодування для зберігання
MODE = 'RGB' # мод читання та запису зображення
MESSAGE = "lishchuk bohdan" # повідомлення для додавання в зображення
RED_CHANNEL_MASK = 0b11111110  # Маска для збереження всіх бітів, крім останнього
BIT_POSITION = 1  # Позиція біта для заміщення (останній біт)
DIR_NAME = 'images' # назва директорії
IMG_NAME = "image_1" # назва фото без розширення

def load_image(image_path):
    """Завантажує зображення та конвертує його у формат RGB, перевіряючи існування файлу."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не знайдено.")
    return Image.open(image_path).convert(MODE)

def save_image(img_arr, output_path):
    """Зберігає зображення у файл."""
    Image.fromarray(img_arr.astype(UINT_8), MODE).save(output_path)

def get_shuffled_indexes(size, seed):
    """Генерує список перемішаних індексів на основі seed."""
    np.random.seed(seed) # встановлення seed для генератора
    indexes = np.arange(size) # утворення індексів з 0 до size - 1
    np.random.shuffle(indexes) # перемішування в довільному форматі
    return indexes

def permute_pixels(img_arr, seed):
    """Виконує псевдовипадкову перестановку пікселів."""
    indexes = get_shuffled_indexes(img_arr.size // 3, seed)  # Генерація переставлених індексів для пікселів
    permuted_pixels = permute_pixels_helper(img_arr, indexes)  # Перестановка пікселів за допомогою перетасованих індексів
    return permuted_pixels.reshape(img_arr.shape)  # Повертаємо зображення в початкову форму


def inverse_permute_pixels(img_arr, seed):
    """Відновлює вихідне зображення після перестановки пікселів."""
    indexes = get_shuffled_indexes(img_arr.size // 3, seed)  # Генерація переставлених індексів для пікселів
    inverse_indexes = np.argsort(indexes)  # Створення зворотних індексів для відновлення початкового порядку
    restored_pixels = permute_pixels_helper(img_arr, inverse_indexes)  # Відновлення пікселів у початковому порядку
    return restored_pixels.reshape(img_arr.shape)  # Повертаємо зображення в початкову форму

def permute_pixels_helper(img_arr, indexes):
    """Спільна частина для перестановки пікселів за індексами."""
    pixel_to_rgb_channels = pixels_to_rbg_channels(img_arr)  # Перетворення зображення в масив пікселів (RGB)
    return pixel_to_rgb_channels[indexes]  # Перестановка пікселів за допомогою заданих індексів

def text_to_binary(text):
    """Конвертує текст у бінарний рядок."""
    binary = ""
    for item in text:
        binary += bin(ord(item))[2:].zfill(8)  # Отримуємо бінарне значення символу та доповнюємо до 8 бітів
    return binary


def binary_to_text(binary_data):
    """Конвертує бінарний рядок у текст."""
    text = ""
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i+8]  # Витягуємо кожен байт (8 бітів)
        text += chr(int(byte, 2))  # Конвертуємо бінарне значення в символ і додаємо до результату
    return text.rstrip('\x00')  # Видаляємо нульові байти (якщо є)


def block_hide(img_arr, secret_data, block_size=8):
    """Приховує секретне повідомлення у блоках зображення."""
    height, width, _ = img_arr.shape  # Отримуємо розміри зображення (висота, ширина, кількість каналів)
    binary_secret_data = text_to_binary(secret_data)  # Перетворюємо секретне повідомлення в бінарний рядок
    padded_binary_data = binary_secret_data + '0' * ((height * width) - len(binary_secret_data))  # Додаємо нулі до бінарного рядка, якщо довжина повідомлення менша за площу зображення
    image_with_hidden_data = img_arr.copy()  # Створюємо копію зображення, щоб не змінювати оригінал
    data_index = 0  # Лічильник для відстеження поточного індексу в бінарному повідомленні
    for row in range(0, height, block_size):  # Проходимо по зображенню блоками розміру `block_size`
        for col in range(0, width, block_size):  # Проходимо по кожному стовпцю в блоках
            if data_index < len(padded_binary_data):  # Перевіряємо, чи є ще бінарні дані для вставки
                red_channel = image_with_hidden_data[row, col, 0]  # Отримуємо перший канал пікселя (червоний)
                image_with_hidden_data[row, col, 0] = replace_least_significant_bit(red_channel, padded_binary_data[data_index]) # Використовуємо нову функцію для заміщення останнього біта
                data_index += 1  # Збільшуємо індекс для наступного біта
    return image_with_hidden_data  # Повертаємо зображення з прихованими даними


def replace_least_significant_bit(red_channel, bit_value):
    """Заміщає останній біт червоного каналу на заданий біт з бінарного повідомлення."""
    # Використовуємо маску червоного каналу для того, щоб залишити всі біти, окрім останнього,
    # а потім додаємо новий біт на місце найменш значущого біта.
    return (red_channel & RED_CHANNEL_MASK) | int(bit_value)


def extract_block_data(img_arr, block_size=8):
    """Витягує приховане повідомлення із зображення."""
    height, weigth, _ = img_arr.shape  # Отримуємо розміри зображення (висота, ширина)
    # Пройдемо по кожному блоку зображення розміру `block_size` і витягнемо останній біт
    # з кожного пікселя в червоному каналі (перший канал зображення).
    binary_data = ''.join(
        str(img_arr[i, j, 0] & 1)  # Отримуємо останній біт червоного каналу
        for i in range(0, height, block_size)  # Ітеруємо по кожному блоку в напрямку висоти
        for j in range(0, weigth, block_size)  # Ітеруємо по кожному блоку в напрямку ширини
    )
    return binary_to_text(binary_data)  # Перетворюємо отриманий бінарний рядок в текст


def apply_palette_substitution(img_arr, color):
    """Зашифровує зображення шляхом заміни палітри кольорів."""
    # Перетворюємо пікселі зображення на канали RGB
    pixels = img_arr.reshape(-1, 3)  # Перетворюємо зображення в масив пікселів (по рядках)
    # Створюємо палітру заміни для кожного унікального пікселя
    # Для кожного пікселя застосовуємо побітову операцію XOR із заданим кольором.
    palette = {}
    for pixel in np.unique(pixels, axis=0):  # Проходимо по кожному унікальному пікселю
        new_pixel = np.bitwise_xor(pixel, color)  # Змінюємо піксель за допомогою XOR
        palette[tuple(pixel)] = tuple(new_pixel)  # Додаємо змінений піксель в палітру
    # Створюємо нове зображення шляхом заміни пікселів на нові за допомогою палітри
    new_img_arr = np.empty_like(img_arr)  # Створюємо порожній масив для нового зображення
    for i in range(img_arr.shape[0]):  # Ітеруємо по кожному ряду зображення
        for j in range(img_arr.shape[1]):  # Ітеруємо по кожному пікселю в ряду
            new_img_arr[i, j] = palette[tuple(img_arr[i, j])]  # Замінюємо піксель на новий з палітри
    return new_img_arr


def reverse_palette_substitution(img_arr, color):
    """Розшифровує зображення після заміни палітри кольорів."""
    pixels = img_arr.reshape(-1, 3)  # Перетворюємо зображення в масив пікселів (по рядках)
    # Створюємо зворотну палітру для відновлення початкового кольору.
    # Для кожного пікселя застосовуємо побітову операцію XOR із кольором.
    reverse_palette = {}
    for pixel in np.unique(pixels, axis=0):  # Проходимо по кожному унікальному пікселю
        original_pixel = np.bitwise_xor(pixel, color)  # Відновлюємо оригінальний піксель за допомогою XOR
        reverse_palette[tuple(pixel)] = tuple(original_pixel)  # Додаємо відновлений піксель в палітру
    # Створюємо нове зображення, використовуючи зворотну палітру для відновлення пікселів
    new_img_arr = np.empty_like(img_arr)  # Створюємо порожній масив для нового зображення
    for i in range(img_arr.shape[0]):  # Ітеруємо по кожному ряду зображення
        for j in range(img_arr.shape[1]):  # Ітеруємо по кожному пікселю в ряду
            new_img_arr[i, j] = reverse_palette.get(tuple(img_arr[i, j]), (0, 0, 0))  # Відновлюємо піксель
    return new_img_arr


def pixels_to_rbg_channels(img_arr):
    return img_arr.reshape(-1, 3)


def swap_pixels_executor(img_arr):
    permuted_image = permute_pixels(img_arr, seed=42)
    save_image(permuted_image, f"{DIR_NAME}/{IMG_NAME}_permuted.jpg")
    restored_image = inverse_permute_pixels(permuted_image, seed=42)
    save_image(restored_image, f"{DIR_NAME}/{IMG_NAME}_restored.jpg")


def block_hide_executor(img_arr):
    hidden_image = block_hide(img_arr, MESSAGE)
    save_image(hidden_image, f"{DIR_NAME}/{IMG_NAME}_hidden.jpg")
    extracted_message = extract_block_data(hidden_image)
    print(f"extracted message [{extracted_message}]")


def change_palette_executor(img_arr):
    secret_color = (99, 99, 99)
    encoded_palette_image = apply_palette_substitution(img_arr, secret_color)
    save_image(encoded_palette_image, f"{DIR_NAME}/{IMG_NAME}_palette_encoded.jpg")
    decoded_palette_image = reverse_palette_substitution(encoded_palette_image, secret_color)
    save_image(decoded_palette_image, f"{DIR_NAME}/{IMG_NAME}_palette_decoded.jpg")


def generate_img_arr():
    image = load_image(f"{DIR_NAME}/{IMG_NAME}.jpg")
    return np.array(image)

if __name__ == '__main__':
    image_array = generate_img_arr()

    # Перестановка пікселів
    swap_pixels_executor(image_array)

    # Блокове приховування
    block_hide_executor(image_array)

    # Заміна палітри
    change_palette_executor(image_array)
