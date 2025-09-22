from skimage.filters import threshold_sauvola
import cv2
import numpy as np
import fitz
import io
from PIL import Image

def preprocess_image_for_ocr(image: np.ndarray, upscale_factor: float = 1.5) -> np.ndarray:
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    binary = (contrast_enhanced > threshold_sauvola(contrast_enhanced, window_size=25)).astype(np.uint8) * 255
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    angle = np.median(angles)
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    processed = cv2.resize(rotated, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LANCZOS4)
    return processed


def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("ppm")))
        images.append(img)
    return images

def process_pdf_for_ocr(input_pdf_path, output_pdf_path, dpi=300):
    images = pdf_to_images(input_pdf_path, dpi=dpi)
    processed_images = []
    for img in images:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = preprocess_image_for_ocr(cv_img)
        processed_pil = Image.fromarray(processed).convert('RGB')
        processed_images.append(processed_pil)

    if processed_images:
        processed_images[0].save(
            output_pdf_path, "PDF", resolution=100.0,
            save_all=True, append_images=processed_images[1:] if len(processed_images) > 1 else []
        )


