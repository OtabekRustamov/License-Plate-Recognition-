import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained Vision Transformer (ViT) model for character recognition
characterRecognition = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=36,  # Set to 36 to match the checkpoint
    ignore_mismatched_sizes=True
)
checkpoint = torch.load('../trained weight/vit_model.pth', weights_only=True)
characterRecognition.load_state_dict(checkpoint)  # Load the model state
characterRecognition.to(device)
characterRecognition.eval()

# Character mapping dictionary for 36 classes (0-9 and A-Z)
dictionary = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
              10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
              20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
              30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

# Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def cnnCharRecognition(img):
    # Preprocess the character image
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    # Predict character
    with torch.no_grad():
        outputs = characterRecognition(img).logits
        predicted_idx = torch.argmax(outputs, dim=1).item()

    # Map prediction to character
    return dictionary[predicted_idx]

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def opencvReadPlate(img):
    charList = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
    edges = auto_canny(thresh_inv)
    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0] * img.shape[1]

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w * h
        non_max_sup = roi_area / img_area

        # Filter out non-character contours
        if (non_max_sup >= 0.015) and (non_max_sup < 0.09):
            if (h > 1.2 * w) and (3 * w >= h):
                # Recognize character
                char_img = img[y:y + h, x:x + w]
                char = cnnCharRecognition(char_img)
                charList.append(char)
                cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)

    # Display the result with bounding boxes using OpenCV
    cv2.imshow("Detected License Plate", img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    licensePlate = "".join(charList)
    return licensePlate

# Load a test image
test_image_path = "../test_data/plate1.PNG"  # Replace with your license plate image path

plate_img = cv2.imread(test_image_path)

# Recognize characters
if plate_img is not None:
    license_text = opencvReadPlate(plate_img)
    print("Recognized License Plate Text:", license_text)
else:
    print("Error: Could not load the image.")
