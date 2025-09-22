import json
import torch
from torchvision import models, transforms
from PIL import Image
import os
import fitz


with open('img_model//class_names.json', 'r') as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load('img_model//best_multi_label_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict_all_classes(image_path, threshold=0.5):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output).cpu().numpy()[0]  # Shape: [NUM_CLASSES]

    predicted_classes = [
        (class_names[idx], float(prob))
        for idx, prob in enumerate(probs)
        if prob >= threshold
    ]
    return predicted_classes





def find_img(pdf_path):
    output_dir = "found_img/pdf_images"
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    det={}
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(output_dir, f"page_{page_num+1}.png")
        pix.save(img_path)
        results =predict_all_classes(img_path)
        if page_num not in det:
            det[page_num]=[results]
        else:
            det[page_num]=det[page_num].append(results)
    return det



