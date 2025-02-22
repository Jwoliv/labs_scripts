import random
from utils import bin_to_text

# Витягування повідомлення з послідовно вбудованого зображення
def extract_message(image, length):
    bin_msg = ''
    rows, cols, channels = image.shape
    for i in range(length * 8):
        row, col = divmod(i, cols)
        if row >= rows:
            break
        pixel_value = image[row, col]
        bin_msg += str(pixel_value[0] & 1)  # Витягуємо LSB із синього каналу
    return bin_to_text(bin_msg)

def extract_message_random(image, length, seed):
    bin_msg = ''
    rows, cols, channels = image.shape
    total_pixels = rows * cols

    random.seed(seed)
    positions = random.sample(range(total_pixels), length * 8)

    for pos in positions:
        row, col = divmod(pos, cols)
        pixel_value = image[row, col]
        bin_msg += str(pixel_value[0] & 1)  # Витягуємо LSB із синього каналу
    return bin_to_text(bin_msg)