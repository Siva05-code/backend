import fitz  # PyMuPDF
import os
from ultralytics import YOLO

model = YOLO(r'img_model\best.pt')



def find_img(pdf_path):
    output_dir = "found_img/pdf_images"
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    detections = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(output_dir, f"page_{page_num+1}.png")
        pix.save(img_path)
        results = model.predict(source=img_path, conf=0.25, imgsz=640, verbose=False)
        found_objects = []
        if results and len(results[0].boxes) > 0:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            for cls_id in class_ids:
                found_objects.append(results[0].names[cls_id])
        detections[page_num + 1] = found_objects
    return detections
