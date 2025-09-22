import fitz
import os
from PIL import Image
from docling.document_converter import DocumentConverter
from parser import extract_document_elements

def extract_and_save_images_from_pdf(pdf_path: str, bbox_data: list, output_folder: str = "found_img"):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder+"/img",exist_ok=True)
    extracted_image_files = []
    try:
        pdf_document = fitz.open(pdf_path)
        print(f"Opened PDF with PyMuPDF: {pdf_path}")
    except Exception as e:
        print(f"Error opening PDF with PyMuPDF: {e}")
        return

    print("\n--- Extracting Images from PDF ---")
    for i, item in enumerate(bbox_data):
        page_no = item['page_no']
        bbox = item['bbox']
        l, t, r, b = float(bbox['l']), float(bbox['t']), float(bbox['r']), float(bbox['b'])

        print(f"\nProcessing Image {i + 1} on Page {page_no + 1}:")  # +1 for human-readable page number
        print(f"  Bounding Box: (l={l:.2f}, t={t:.2f}, r={r:.2f}, b={b:.2f})")

        try:
            if page_no >= len(pdf_document):
                print(f"  Error: Page {page_no + 1} does not exist in the PDF. Skipping.")
                continue

            fitz_page = pdf_document[page_no]

            page_width = fitz_page.rect.width
            page_height = fitz_page.rect.height

            x0 = max(0, l)
            y0 = max(0, page_height - t)
            x1 = min(page_width, r)
            y1 = min(page_height, page_height - b)

            if x0 >= x1 or y0 >= y1:
                print(
                    f"  Warning: Invalid or degenerate bounding box coordinates after conversion for image {i + 1}. Skipping.")
                print(f"  Converted Coords: (x0={x0:.2f}, y0={y0:.2f}, x1={x1:.2f}, y1={y1:.2f})")
                continue

            fitz_rect = fitz.Rect(x0, y0, x1, y1)

            matrix = fitz.Matrix(1.5, 1.5)
            pixmap = fitz_page.get_pixmap(matrix=matrix, clip=fitz_rect)

            if pixmap:
                image_filename = f"image_p{page_no + 1}_x{int(l)}_y{int(t)}_w{int(r - l)}_h{int(b - t)}.png"
                image_path = os.path.join(output_folder+ "/img" , image_filename)
                pixmap.save(image_path)
                extracted_image_files.append(image_path)
                print(f"  Saved image to: {image_path}")
                pixmap = None  # Release memory
            else:
                print(f"  Could not create pixmap for image {i + 1}.")

        except Exception as e:
            print(f"  Error extracting image {i + 1} with PyMuPDF: {e}")

    pdf_document.close()

    if extracted_image_files:
        print("\n--- Creating PDF of Extracted Images ---")
        try:
            output_pdf_path = os.path.join(output_folder, "extracted_images_summary.pdf")
            new_pdf = fitz.open()

            for img_file in extracted_image_files:
                img = Image.open(img_file)
                img_width, img_height = img.size

                page = new_pdf.new_page(width=img_width + 40, height=img_height + 40)  # Add some padding

                page.insert_image(fitz.Rect(20, 20, 20 + img_width, 20 + img_height), filename=img_file)

            new_pdf.save(output_pdf_path)
            new_pdf.close()
            print(f"Successfully created PDF of extracted images: {output_pdf_path}")
        except Exception as e:
            print(f"Error creating PDF of extracted images: {e}")
    else:
        print("\nNo images were extracted to create a summary PDF.")


def write_text_to_pdf_from_data(text_data: list, output_pdf_path: str, bbox_buffer: float = 3.0, text_color=(0.8, 0, 0)):
    doc = None
    try:
        doc = fitz.open()

        default_page_width = 595.276
        default_page_height = 841.89

        max_page_needed = -1
        if text_data:
            max_page_needed = max(item['page_no'] for item in text_data)

        current_page_count = doc.page_count
        if current_page_count <= max_page_needed:
            for _ in range(current_page_count, max_page_needed + 1):
                if current_page_count > 0:
                    first_page_rect = doc[0].rect
                    doc.new_page(width=first_page_rect.width, height=first_page_rect.height)
                else:
                    doc.new_page(width=default_page_width, height=default_page_height)
            print(f"Added {max_page_needed - current_page_count + 1} new pages to the document.")

        print(f"\n--- Writing Text to PDF with {bbox_buffer}pt Buffer ---")

        for i, item in enumerate(text_data):
            page_no = item['page_no']
            bbox = item['bbox']
            text_to_write = item['text']

            l, t, r, b = float(bbox['l']), float(bbox['t']), float(bbox['r']), float(bbox['b'])

            print(f"\nProcessing Text Item {i + 1} on Page {page_no + 1}:")
            print(f"  Original Text: '{text_to_write}'")
            print(f"  Original BBox: (l={l:.2f}, t={t:.2f}, r={r:.2f}, b={b:.2f})")

            try:
                if page_no >= len(doc):
                    print(
                        f"  Error: Page {page_no + 1} still does not exist despite attempts to create it. Skipping text insertion.")
                    continue

                page = doc[page_no]

                page_width = page.rect.width
                page_height = page.rect.height

                fitz_x0 = l
                fitz_y0 = page_height - t
                fitz_x1 = r
                fitz_y1 = page_height - b

                buffered_x0 = max(0, fitz_x0 - bbox_buffer)
                buffered_y0 = max(0, fitz_y0 - bbox_buffer)
                buffered_x1 = min(page_width, fitz_x1 + bbox_buffer)
                buffered_y1 = min(page_height, fitz_y1 + bbox_buffer)

                text_rect = fitz.Rect(buffered_x0, buffered_y0, buffered_x1, buffered_y1)

                if text_rect.is_empty or text_rect.width <= 0 or text_rect.height <= 0:
                    print(
                        f"  Warning: Invalid or degenerate text rectangle for item {i + 1} after buffering ({text_rect}). Skipping text insertion.")
                    continue

                page.insert_textbox(
                    text_rect,
                    text_to_write,
                    fontsize=9,
                    fontname="helv",
                    set_simple=True,
                    align=fitz.TEXT_ALIGN_LEFT,
                    color=text_color
                )
                print(f"  Wrote text '{text_to_write}' into buffered Rect {text_rect}.")

            except Exception as e:
                print(f"  Error writing text for item {i + 1}: {e}")

        doc.save(output_pdf_path)
        print(f"\nSuccessfully created PDF with text: {output_pdf_path}")

    except Exception as e:
        print(f"An unexpected error occurred during PDF processing: {e}")
    finally:
        if doc:
            doc.close()

