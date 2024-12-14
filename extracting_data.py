import fitz  # PyMuPDF
import os
import cv2
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

# Path to your PDF file
print("amk")

pdf_path = 'D:\TheoryOfComputation.pdf'  # Replace with the actual path

# Define output folders
image_folder = 'extracted_images'
vector_folder = 'vector_graphics'
text_free_folder = 'text_free_images'
shape_folder = 'shapes'

# Create output folders if they don't exist
os.makedirs(image_folder, exist_ok=True)
os.makedirs(vector_folder, exist_ok=True)
os.makedirs(text_free_folder, exist_ok=True)
os.makedirs(shape_folder, exist_ok=True)

# Open the PDF
pdf = fitz.open(pdf_path)

# Iterate through all pages in the PDF
for page_number in range(len(pdf)):
    page = pdf.load_page(page_number)

    # Extract raster images (JPEG, PNG)
    images_on_page = page.get_images(full=True)

    if images_on_page:
        for image_index, image in enumerate(images_on_page):
            img_reference = image[0]
            base_image = pdf.extract_image(img_reference)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"page_{page_number + 1}_image_{image_index + 1}.{image_ext}"
            image_path = os.path.join(image_folder, image_filename)

            with open(image_path, "wb",encoding='utf-8') as image_file:
                image_file.write(image_bytes)
            print(f"Image saved: {image_path}")
    else:
        # Save the page as a full-page image (rasterized)
        pix = page.get_pixmap()
        image_filename = f"page_{page_number + 1}_full_page.png"
        image_path = os.path.join(image_folder, image_filename)
        pix.save(image_path)
        print(f"Full page image saved: {image_path}")

    # Extract vector graphics (SVG format)
    vector_text = page.get_text("svg")
    if vector_text:
        vector_filename = f"page_{page_number + 1}_vector.svg"
        vector_path = os.path.join(vector_folder, vector_filename)

        with open(vector_path, 'w',encoding='utf-8') as vector_file:
            vector_file.write(vector_text)
        print(f"Vector content saved: {vector_path}")

# Close the PDF file
pdf.close()

# OCR and Text Masking
ocr = PaddleOCR()
for image_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, image_name)
    img = cv2.imread(img_path)

    results = ocr.ocr(img_path)

    for line in results[0]:
        box = line[0]
        box = [(int(x), int(y)) for x, y in box]
        cv2.fillPoly(img, [np.array(box, dtype=np.int32)], (255, 255, 255))

    output_path = os.path.join(text_free_folder, image_name)
    cv2.imwrite(output_path, img)
    print(f"Image without text saved as '{output_path}'")

# Shape Extraction
for image_name in os.listdir(text_free_folder):
    img_path = os.path.join(text_free_folder, image_name)
    image = cv2.imread(img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    for i, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_shape = image[y:y+h, x:x+w]

        output_path = os.path.join(shape_folder, f"shape_{i+1}_{image_name}")
        cv2.imwrite(output_path, cropped_shape)

print("Image and vector extraction, OCR, and shape extraction completed.")