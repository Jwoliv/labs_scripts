import numpy as np
from PIL import Image

def embed_data(image_path_param, secret_data_param, output_path_param):
    # Завантаження зображення
    image = Image.open(image_path_param)
    img_data = np.array(image)

    # Перетворення секретних даних у бінарний вигляд
    secret_data_bin = ''.join(format(ord(i), '08b') for i in secret_data_param)
    data_len = len(secret_data_bin)

    # Перевірка, чи є місце для вбудовування
    if data_len > img_data.size:
        raise ValueError("Не вистачає місця для вбудовування всіх даних.")

    # Вбудовування бітів секретної інформації в синій канал
    idx = 0
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            # Отримуємо піксель
            pixel = img_data[i, j]
            blue_value = pixel[2]  # Синій канал (RGB)

            if idx < data_len:
                # Вбудовуємо біт у найменш значущий біт синього каналу
                blue_value = (blue_value & 0xFE) | int(secret_data_bin[idx])
                img_data[i, j][2] = blue_value
                idx += 1

    # Збереження модифікованого зображення
    output_image = Image.fromarray(img_data)
    output_image.save(output_path_param)
    print("Секретні дані вбудовано в зображення.")

def extract_data(image_path_param, data_length_param):
    # Завантаження зображення
    image = Image.open(image_path_param)
    img_data = np.array(image)

    secret_data_bin = ""
    idx = 0
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            # Отримуємо піксель
            pixel = img_data[i, j]
            blue_value = pixel[2]  # Синій канал (RGB)
            # Витягуємо біт з найменш значущого біта синього каналу
            secret_data_bin += str(blue_value & 1)
            idx += 1
            if idx >= data_length_param * 8:
                break
        if idx >= data_length_param * 8:
            break

    # Перетворення бінарного рядка в символи
    result_secret_data = ''.join(chr(int(secret_data_bin[i:i+8], 2)) for i in range(0, len(secret_data_bin), 8))
    return result_secret_data

if __name__ == '__main__':
    image_path = "images/image_1.jpg"
    output_path = "output_image.png"
    secret_data = "lishchuk bohdan"

    embed_data(image_path, secret_data, output_path)
    extracted_data = extract_data(output_path, len(secret_data))
    print("Витягнуті дані:", extracted_data)
