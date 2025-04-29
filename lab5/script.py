import cv2
import numpy as np


def embed_watermark_dct(container_img, watermark_img, strength=10):
    h, w = container_img.shape
    watermark_resized = cv2.resize(watermark_img, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
    watermark_bin = (watermark_resized > 128).astype(np.uint8)
    watermarked_img = np.zeros_like(container_img, dtype=np.float32)
    container_img = container_img.astype(np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = container_img[i:i + 8, j:j + 8]
            if block.shape[0] < 8 or block.shape[1] < 8:
                continue

            dct_block = cv2.dct(block)

            wm_i = i // 8
            wm_j = j // 8
            if wm_i < watermark_bin.shape[0] and wm_j < watermark_bin.shape[1]:
                bit = watermark_bin[wm_i, wm_j]
                if bit == 1:
                    dct_block[4, 4] += strength
                else:
                    dct_block[4, 4] -= strength

            idct_block = cv2.idct(dct_block)
            watermarked_img[i:i + 8, j:j + 8] = idct_block

    return np.clip(watermarked_img, 0, 255).astype(np.uint8)


def extract_watermark_dct(watermarked_img, container_img, wm_shape, strength=10):
    h, w = container_img.shape
    extracted = np.zeros(wm_shape, dtype=np.uint8)

    container_img = container_img.astype(np.float32)
    watermarked_img = watermarked_img.astype(np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block_orig = container_img[i:i + 8, j:j + 8]
            block_wm = watermarked_img[i:i + 8, j:j + 8]

            if block_orig.shape[0] < 8 or block_orig.shape[1] < 8:
                continue

            dct_orig = cv2.dct(block_orig)
            dct_wm = cv2.dct(block_wm)

            delta = dct_wm[4, 4] - dct_orig[4, 4]

            wm_i = i // 8
            wm_j = j // 8
            if wm_i < wm_shape[0] and wm_j < wm_shape[1]:
                extracted[wm_i, wm_j] = 255 if delta > 0 else 0

    return extracted


container_path = "images/image_1.jpg"
watermark_path = "images/watermark.png"
visible_output_path = "images/watermarked_dct.png"
extracted_output_path = "images/extracted_dct.png"

container = cv2.imread(container_path, cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

if container is None or watermark is None:
    raise FileNotFoundError("❌ Check that 'image_1.jpg' and 'watermark.png' exist in the 'images/' folder.")

watermarked_dct = embed_watermark_dct(container, watermark, strength=150)
cv2.imwrite(visible_output_path, watermarked_dct)

extracted_dct = extract_watermark_dct(watermarked_dct, container, (container.shape[0] // 8, container.shape[1] // 8), strength=150)
cv2.imwrite(extracted_output_path, extracted_dct)

print("✅ Watermark embedded using DCT and extracted using Hsu-Wu method.")
