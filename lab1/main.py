from datetime import datetime
from encoder import embed_sequential, embed_random
from decoder import extract_message, extract_message_random
from utils import read_jpg, write_jpg, plot_histogram, mse

# Основне виконання
if __name__ == '__main__':
    image = read_jpg("images/image_2.jpg")  # Читаємо зображення з файлу
    message = """
        Lorem Ipsum is simply dummy text of the printing and typesetting industry.
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s...
        """  # Повідомлення для вбудовування

    # --- Кодування ---
    image_seq = embed_sequential(image, message)
    write_jpg(f"./images/image_seq_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg", image_seq)

    key = 340698234968  # Ключ для випадкового вбудовування
    image_rand = embed_random(image, message, key)
    write_jpg(f"./images/image_random_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg", image_rand)

    # --- Декодування ---
    retrieved_msg_seq = extract_message(image_seq, len(message))
    print("Retrieved Message (Sequential):", retrieved_msg_seq)

    retrieved_msg_rand = extract_message_random(image_rand, len(message), key)
    print("Retrieved Message (Random):", retrieved_msg_rand)

    # --- Аналіз ---
    plot_histogram(image, image_seq)  # Порівняння гістрограм
    print("MSE (Sequential):", mse(image, image_seq))
    print("MSE (Random):", mse(image, image_rand))
