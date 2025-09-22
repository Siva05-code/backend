import os
import json
import fitz
import yaml
import tempfile
import time
import random
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

# LangChain imports for replacement generation
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from custom_wrapper import OpenRouterChat
from pydantic import BaseModel, Field

# Import your existing modules
from enhanceimg import process_pdf_for_ocr
from parser import extract_document_elements
from img_ext import extract_and_save_images_from_pdf, write_text_to_pdf_from_data
from pii_agent import execute_pii
from img_model.predict_lables import find_img
from docling.document_converter import DocumentConverter

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class PIIOutput(BaseModel):
    """Pydantic model for PII replacement output"""
    pii_types: List[str] = Field(default_factory=list)
    pii_types_replacement: List[str] = Field(default_factory=list)


class SecureRedactionPolicyManager:
    """Enhanced policy manager with global policies (not entity-dependent)"""

    def __init__(self, policy_path: str = None):
        self.policies = self._create_default_policies()
        if policy_path and os.path.exists(policy_path):
            self._load_policies(policy_path)

    def _create_default_policies(self) -> Dict[str, Any]:
        return {
            'global_settings': {
                'text_redaction_mode': 'dummy_replacement',  # 'dummy_replacement' or 'anonymize'
                'visual_redaction_mode': 'text_box',  # 'text_box' or 'image'
                'audit_logging': True,
                'secure_processing': True,
                'memory_only': True,
                'create_overlay_pdf': True
            }
        }

    def _load_policies(self, policy_path: str):
        try:
            with open(policy_path, 'r') as f:
                loaded_policies = yaml.safe_load(f)
                self.policies.update(loaded_policies)
        except Exception as e:
            print(f"Warning: Could not load policies from {policy_path}: {e}")

    def get_text_redaction_mode(self) -> str:
        """Get global text redaction mode"""
        return self.policies.get('global_settings', {}).get('text_redaction_mode', 'dummy_replacement')

    def get_visual_redaction_mode(self) -> str:
        """Get global visual redaction mode"""
        return self.policies.get('global_settings', {}).get('visual_redaction_mode', 'text_box')

    def should_create_overlay(self) -> bool:
        return self.policies.get('global_settings', {}).get('create_overlay_pdf', True)


class LangChainDummyDataGenerator:
    """LangChain-based dummy data generator using LLM"""

    def __init__(self):
        self._dummy_cache = {}  # In-memory cache for session
        self._setup_langchain()

    def _setup_langchain(self):
        """Setup LangChain components"""
        try:
            self.llm = OpenRouterChat(
                api_key=OPENROUTER_API_KEY,
                model="openai/gpt-3.5-turbo",
                temperature=0,
                max_tokens=1024
            )

            self.parser = PydanticOutputParser(pydantic_object=PIIOutput)

            self.prompt = ChatPromptTemplate.from_template("""
You are a Fake information generator for the following given types in the list.
List of PII Types:
{pii_types}

Generate realistic but fake replacements for each type that would be appropriate for document redaction. 
Make sure the replacements are contextually appropriate and maintain similar formatting.

For example:
- Names: Generate realistic full names
- Addresses: Generate complete addresses with street, city, state, zip
- Phone numbers: Generate in standard format (XXX) XXX-XXXX
- Emails: Generate realistic email addresses
- SSN: Generate in XXX-XX-XXXX format
- Dates: Generate in MM/DD/YYYY format

Output strictly as JSON in the following structure:

{{
  "pii_types": ["Name", "Birthday", ...],
  "pii_types_replacement": ["John Smith", "01/15/1985", ...]
}}
""")

            self.chain = (
                    {"pii_types": RunnablePassthrough()}
                    | self.prompt
                    | self.llm
                    | self.parser
            )

        except Exception as e:
            print(f"Warning: Could not setup LangChain components: {e}")
            self.llm = None
            self.chain = None

    def generate_dummy_data(self, pii_types: List[str]) -> Dict[str, str]:
        """Generate dummy data for a list of PII types"""
        if not pii_types:
            return {}

        # Check cache first
        cache_key = ",".join(sorted(pii_types))
        if cache_key in self._dummy_cache:
            return self._dummy_cache[cache_key]

        try:
            if self.chain is None:
                return self._get_fallback_data(pii_types)

            # Use LangChain to generate replacements
            result = self.chain.invoke(", ".join(pii_types))
            result_dict = result.dict()

            # Create mapping
            replacement_map = {}
            for i, pii_type in enumerate(result_dict.get('pii_types', [])):
                if i < len(result_dict.get('pii_types_replacement', [])):
                    replacement_map[pii_type] = result_dict['pii_types_replacement'][i]

            # Cache the result
            self._dummy_cache[cache_key] = replacement_map
            return replacement_map

        except Exception as e:
            print(f"Error generating dummy data with LangChain: {e}")
            return self._get_fallback_data(pii_types)

    def _get_fallback_data(self, pii_types: List[str]) -> Dict[str, str]:
        """Fallback dummy data when LangChain fails"""
        fallbacks = {
            'Name': 'John Smith',
            'Address': '123 Main Street, Anytown, State 12345',
            'Birthday': '01/15/1985',
            'Email': 'john.smith@example.com',
            'Phone': '(555) 123-4567',
            'SSN': '123-45-6789',
            'Passport': 'A12345678',
            'Credit card': '1234-5678-9012-3456',
            'Age': '35',
            'Gender': 'Non-binary',
            'Race': 'Mixed',
            'Location': 'Sample City',
            'Medical Condition': 'General wellness check',
            'Medication': 'Over-the-counter supplement',
            'Doctor Name': 'Dr. Smith',
            'Hospital Name': 'General Medical Center',
            'Medical Record Number': 'MRN123456',
            'Health Plan Beneficiary Number': 'HPN987654',
            'Account Number': 'ACC123456789',
            'Web URL': 'https://example.com',
            'IP Address': '192.168.1.1'
        }

        return {pii_type: fallbacks.get(pii_type, f'[Dummy {pii_type}]') for pii_type in pii_types}

    def get_replacement_for_type(self, pii_type: str, original_text: str = "") -> str:
        """Get a single replacement for a specific PII type"""
        replacement_map = self.generate_dummy_data([pii_type])
        return replacement_map.get(pii_type, f'[Dummy {pii_type}]')


class SecureInfoRedactionPipeline:
    """Main secure redaction pipeline for PII and PHI"""

    def __init__(self, input_pdf: str, policy_manager: SecureRedactionPolicyManager = None):
        self.input_pdf = input_pdf
        self.policy_manager = policy_manager or SecureRedactionPolicyManager()
        self.dummy_generator = LangChainDummyDataGenerator()
        self.processing_log = {
            'timestamp': datetime.now().isoformat(),
            'input_pdf': input_pdf,
            'redactions': [],
            'metrics': defaultdict(int)
        }
        self.redaction_spans = []  # Store redaction spans for overlay creation

    def _docling_bbox_to_fitz(self, bb, page_height):
        """Convert Docling bbox to fitz coordinates with proper transformation"""
        if isinstance(bb, dict):
            l, t, r, b = bb['l'], bb['t'], bb['r'], bb['b']
        else:
            l, t, r, b = bb[0], bb[1], bb[2], bb[3]

        # Proper coordinate transformation for PDF coordinate system
        x0 = l
        y0 = page_height - b  # Bottom of bbox in PDF coordinates
        x1 = r
        y1 = page_height - t  # Top of bbox in PDF coordinates

        return fitz.Rect(x0, y0, x1, y1).normalize()

    def _get_img_path(self, name, save_dir="downloaded_images"):
        """Download an image from Google search using a more reliable approach"""
        os.makedirs(save_dir, exist_ok=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        query = f"{name}"
        search_url = f"https://www.google.com/search?tbm=isch&q={quote_plus(query)}"

        try:
            response = requests.get(search_url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try to find image URLs in the page source
            script_tags = soup.find_all('script')

            # Look for image URLs in the script content
            image_urls = []
            for script in script_tags:
                if script.string:
                    urls = re.findall(r'\"(https?://[^\"]+\.(?:jpg|jpeg|png|gif))\"', script.string)
                    image_urls.extend(urls)

            # If we didn't find URLs in scripts, try a different approach
            if not image_urls:
                images = soup.find_all('img')
                for img in images:
                    data_src = img.get('data-src')
                    if data_src and data_src.startswith('http'):
                        image_urls.append(data_src)

            # Remove duplicates
            image_urls = list(set(image_urls))

            # Try to download the first valid image
            for img_url in image_urls[:5]:  # Try first 5 URLs
                try:
                    time.sleep(random.uniform(0.5, 1.5))
                    img_data = requests.get(img_url, headers=headers, timeout=15).content

                    # Check if the image data is valid
                    if len(img_data) > 1000:  # Minimum size check
                        safe_name = re.sub(r'[^\w\-_\. ]', '_', name)
                        image_path = os.path.join(save_dir, f"{safe_name}.jpg")

                        with open(image_path, 'wb') as handler:
                            handler.write(img_data)

                        print(f"   Downloaded image for '{name}' from {img_url[:50]}...")
                        return image_path

                except Exception as e:
                    print(f"   Error downloading image from {img_url[:50]}...: {e}")
                    continue

            print(f"   No valid image found for '{name}' after trying {len(image_urls)} URLs")
            return None

        except Exception as e:
            print(f"   Error searching for image '{name}': {e}")
            return None

    def _get_simplified_img_path(self, name, save_dir="downloaded_images"):
        """Alternative approach: Use a simpler search term for better results"""
        # Simplify the search term - remove "closeup", "zoom", etc.
        simplified_name = name.lower()
        for term in ["closeup", "zoom", "scan", "sample"]:
            simplified_name = simplified_name.replace(term, "")

        simplified_name = simplified_name.strip()
        if not simplified_name:
            simplified_name = name  # Fallback to original name

        print(f"   Simplified search term: '{simplified_name}' (from '{name}')")
        return self._get_img_path(simplified_name, save_dir)

    def _insert_image_in_bbox(self, page, rect, image_path):
        """Insert an image into the specified rectangle on the PDF page"""
        try:
            # Open the image
            img = fitz.open(image_path)
            # Get the pixmap of the image
            pix = img[0].get_pixmap()

            # Calculate scaling factors to fit the image within the rectangle
            img_width = pix.width
            img_height = pix.height
            rect_width = rect.width
            rect_height = rect.height

            # Calculate scaling factors
            scale_x = rect_width / img_width
            scale_y = rect_height / img_height

            # Use the smaller scaling factor to maintain aspect ratio
            scale = min(scale_x, scale_y)

            # Calculate new dimensions
            new_width = img_width * scale
            new_height = img_height * scale

            # Calculate position to center the image
            x_center = rect.x0 + (rect_width - new_width) / 2
            y_center = rect.y0 + (rect_height - new_height) / 2

            # Create a new rectangle for the image
            img_rect = fitz.Rect(x_center, y_center, x_center + new_width, y_center + new_height)

            # Insert the image
            page.insert_image(img_rect, pixmap=pix)

            img.close()
            return True

        except Exception as e:
            print(f"   Error inserting image: {e}")
            return False

    def _safe_file_cleanup(self, file_path: str, max_retries: int = 5):
        """Safely cleanup temporary files with retries"""
        if not file_path or not os.path.exists(file_path):
            return

        for attempt in range(max_retries):
            try:
                # Try to close any file handles that might be open
                import gc
                gc.collect()  # Force garbage collection
                time.sleep(0.1 * (attempt + 1))  # Increasing delay

                os.unlink(file_path)
                return  # Success
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Cleanup attempt {attempt + 1} failed for {file_path}, retrying...")
                    time.sleep(0.3 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Warning: Could not delete {file_path} after {max_retries} attempts: {e}")
                    # As a last resort, try to move the file to a different location
                    try:
                        backup_path = file_path + f".backup_{int(time.time())}"
                        os.rename(file_path, backup_path)
                        print(f"Moved problematic file to: {backup_path}")
                    except:
                        pass
        """Safely cleanup temporary files with retries"""
        if not file_path or not os.path.exists(file_path):
            return

        for attempt in range(max_retries):
            try:
                # Try to close any file handles that might be open
                import gc
                gc.collect()  # Force garbage collection
                time.sleep(0.1 * (attempt + 1))  # Increasing delay

                os.unlink(file_path)
                return  # Success
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Cleanup attempt {attempt + 1} failed for {file_path}, retrying...")
                    time.sleep(0.3 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Warning: Could not delete {file_path} after {max_retries} attempts: {e}")
                    # As a last resort, try to move the file to a different location
                    try:
                        backup_path = file_path + f".backup_{int(time.time())}"
                        os.rename(file_path, backup_path)
                        print(f"Moved problematic file to: {backup_path}")
                    except:
                        pass

    def create_overlay_pdf(self, overlay_output: str = "redaction_overlay.pdf"):
        """Create overlay PDF showing redaction locations with correct positioning"""
        if not self.redaction_spans:
            print("ℹ️ No redactions to create overlay for")
            return

        doc = None
        try:
            doc = fitz.open(self.input_pdf)
            print(f"\n=== Creating Overlay PDF ===")

            for span in self.redaction_spans:
                page_num = span["page"]
                if page_num >= len(doc):
                    continue

                page = doc[page_num]
                bbox = span["bbox"]

                # Convert bbox to proper fitz coordinates
                if isinstance(bbox, dict):
                    rect = self._docling_bbox_to_fitz(bbox, page.rect.height)
                elif isinstance(bbox, list) and len(bbox) == 4:
                    # If it's already in [x0, y0, x1, y1] format
                    rect = fitz.Rect(bbox)
                else:
                    print(f"Warning: Invalid bbox format for span: {span}")
                    continue

                # Ensure rect is valid
                if rect.is_empty or rect.is_infinite:
                    print(f"Warning: Invalid rect for span: {span}")
                    continue

                # Color coding based on global policy
                text_mode = self.policy_manager.get_text_redaction_mode()
                if text_mode == "dummy_replacement":
                    color = [0, 1, 0]  # Green
                else:
                    color = [1, 0, 0]  # Red (anonymize)

                try:
                    highlight = page.add_highlight_annot(rect)
                    highlight.set_colors(stroke=color)
                    highlight.set_info(content=f"{span['type']} - {text_mode}")
                    highlight.update()
                except Exception as e:
                    print(f"Warning: Could not add highlight for span {span['span_id']}: {e}")

            doc.save(overlay_output)
            print(f"✅ Overlay PDF created: {overlay_output}")
            return overlay_output

        except Exception as e:
            print(f"Error creating overlay PDF: {e}")
            return None
        finally:
            if doc:
                doc.close()

    def _enhanced_pii_phi_detection_per_page(self, txt_blocks: List[Dict]) -> Dict[str, Any]:
        """Enhanced PII/PHI detection processed per page"""
        all_entities = []
        all_types = []

        # Group text blocks by page
        pages_data = defaultdict(list)
        for block in txt_blocks:
            page_num = block.get('page_no', 0)
            pages_data[page_num].append(block)

        # Process each page separately
        for page_num, page_blocks in pages_data.items():
            print(f"   Processing page {page_num + 1} for PII/PHI...")

            # Use existing PII detection for this page
            try:
                pii_result = execute_pii(page_blocks)

                if pii_result and pii_result.get('pii_entities'):
                    page_entities = pii_result.get('pii_entities', [])
                    page_types = pii_result.get('pii_types', [])

                    all_entities.extend(page_entities)
                    all_types.extend(page_types)

                    print(f"      Found {len(page_entities)} entities on page {page_num + 1}")
            except Exception as e:
                print(f"      Error processing page {page_num + 1}: {e}")
                continue

            # Extend with PHI patterns for this page
            page_text = "\n".join([item['text'] for item in page_blocks])

            phi_patterns = {
                'Medical Record Number': r'MRN[:\s]*(\d{6,})',
                'Health Plan Beneficiary Number': r'Member ID[:\s]*(\w{8,})',
                'Medical Condition': r'(?:diagnosed with|suffers from|condition:)\s*([A-Za-z\s]{5,30})',
                'Medication': r'(?:prescribed|taking|medication:)\s*([A-Za-z]{4,20})',
                'Doctor Name': r'Dr\.?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                'Hospital Name': r'([A-Z][a-z\s]+(?:Hospital|Medical Center|Clinic))'
            }

            for phi_type, pattern in phi_patterns.items():
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                for match in matches:
                    all_entities.append(match)
                    all_types.append(phi_type)

        return {
            'pii_entities': all_entities,
            'pii_types': all_types
        }

    def _map_text_spans_to_pdf(self, text_pdf_path: str, pii_entities: List[str],
                               pii_types: List[str]) -> List[Dict]:
        """Map detected PII/PHI to PDF coordinates"""
        doc = None
        try:
            doc = fitz.open(text_pdf_path)
            results = []
            type_counters = defaultdict(int)

            for page_num, page in enumerate(doc):
                for ent, ent_type in zip(pii_entities, pii_types):
                    matches = page.search_for(ent)
                    type_counters[ent_type] += 1

                    for match_idx, bbox in enumerate(matches):
                        span_data = {
                            "page": page_num,
                            "text": ent,
                            "type": ent_type,
                            "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                            "span_id": f"page_{page_num}_{ent_type}_{type_counters[ent_type]}_{match_idx}",
                            "occurrence": type_counters[ent_type]
                        }
                        results.append(span_data)
                        # Store for overlay creation
                        self.redaction_spans.append(span_data)

            return results
        finally:
            if doc:
                doc.close()

    def _get_replacement_text(self, pii_type: str, original_text: str,
                              occurrence: int) -> str:
        """Get replacement text based on global policy"""
        text_mode = self.policy_manager.get_text_redaction_mode()

        if text_mode == "anonymize":
            return f"[{pii_type}_{occurrence}]"
        elif text_mode == "dummy_replacement":
            return self.dummy_generator.get_replacement_for_type(pii_type, original_text)
        else:
            return f"[REDACTED_{pii_type}]"

    def _redact_text_in_memory(self, text_pdf_data: bytes, pii_spans: List[Dict]) -> bytes:
        """Redact text in PDF data without saving intermediate files"""
        temp_file_path = None
        doc = None

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(text_pdf_data)
                temp_file.flush()
                temp_file_path = temp_file.name

            doc = fitz.open(temp_file_path)

            for span in pii_spans:
                page = doc[span["page"]]
                bbox = fitz.Rect(span["bbox"])

                replacement = self._get_replacement_text(
                    span["type"], span["text"], span["occurrence"]
                )

                page.add_redact_annot(bbox, text=replacement)

                self.processing_log['redactions'].append({
                    'page': span['page'] + 1,
                    'type': span['type'],
                    'original': span['text'],
                    'replacement': replacement,
                    'action': self.policy_manager.get_text_redaction_mode()
                })

            # Apply redactions
            for page in doc:
                page.apply_redactions()

            # Get redacted PDF as bytes
            redacted_data = doc.write()
            return redacted_data

        finally:
            if doc:
                doc.close()
            if temp_file_path:
                self._safe_file_cleanup(temp_file_path)

    def _create_image_summary_pdf_robust(self, img_data: List[Dict], output_path: str):
        """Create summary PDF for image classification with robust file handling"""
        doc = None
        summary_doc = None

        try:
            doc = fitz.open(self.input_pdf)
            summary_doc = fitz.open()

            for img_info in img_data:
                page_num = img_info['page_no']
                bbox = img_info['bbox']

                if page_num < len(doc):
                    source_page = doc[page_num]

                    # Convert bbox to fitz rect with proper coordinates
                    if isinstance(bbox, dict):
                        rect = self._docling_bbox_to_fitz(bbox, source_page.rect.height)
                    else:
                        rect = fitz.Rect(bbox)

                    # Create new page in summary
                    summary_page = summary_doc.new_page(width=rect.width, height=rect.height)

                    # Copy image area
                    summary_page.show_pdf_page(
                        summary_page.rect,
                        doc,
                        page_num,
                        clip=rect
                    )

            # Close source document completely
            doc.close()
            doc = None

            # Use a completely different approach: write to bytes first, then to file
            # This avoids PyMuPDF's internal file replacement logic
            pdf_bytes = summary_doc.write()
            summary_doc.close()
            summary_doc = None

            # Force garbage collection to release any remaining handles
            import gc
            gc.collect()
            time.sleep(0.2)

            # Remove existing file if it exists
            if os.path.exists(output_path):
                for retry in range(5):
                    try:
                        os.remove(output_path)
                        break
                    except (PermissionError, OSError):
                        time.sleep(0.2 * (retry + 1))
                        gc.collect()

            # Write the PDF bytes to the output file
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

            print(f"   Successfully created image summary PDF with {len(img_data)} images")

        except Exception as e:
            print(f"Error creating image summary PDF: {e}")
            # Create a minimal fallback PDF that can still be used for classification
            try:
                if doc:
                    doc.close()
                if summary_doc:
                    summary_doc.close()

                # Create a fallback with a simple page
                fallback_doc = fitz.open()
                fallback_page = fallback_doc.new_page()
                fallback_page.insert_text((50, 50), "Fallback for image processing", fontsize=12)

                pdf_bytes = fallback_doc.write()
                fallback_doc.close()

                # Clean up existing file
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass

                with open(output_path, 'wb') as f:
                    f.write(pdf_bytes)

            except Exception as fallback_error:
                print(f"Fallback PDF creation failed: {fallback_error}")
                raise Exception(f"Could not create image summary PDF: {e}")
        finally:
            if doc:
                try:
                    doc.close()
                except:
                    pass
            if summary_doc:
                try:
                    summary_doc.close()
                except:
                    pass

    def _process_visual_elements_with_classification(self, redacted_pdf_data: bytes, img_data: List[Dict]) -> bytes:
        """Process visual elements with full image classification"""
        if not img_data:
            return redacted_pdf_data

        temp_input_path = None
        temp_output_path = None
        img_summary_path = None

        try:
            # Create unique temporary file names with process ID to avoid conflicts
            temp_dir = tempfile.gettempdir()
            pid = os.getpid()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            temp_input_path = os.path.join(temp_dir, f"redact_input_{pid}_{timestamp}.pdf")
            temp_output_path = os.path.join(temp_dir, f"redact_output_{pid}_{timestamp}.pdf")
            img_summary_path = os.path.join(temp_dir, f"redact_summary_{pid}_{timestamp}.pdf")

            print(f"   Creating temporary files for image processing...")

            # Write input file
            with open(temp_input_path, 'wb') as f:
                f.write(redacted_pdf_data)

            # Create image summary PDF for classification
            print(f"   Creating image summary PDF for classification...")
            self._create_image_summary_pdf_robust(img_data, img_summary_path)

            # Verify the summary file was created
            if not os.path.exists(img_summary_path):
                raise Exception("Image summary PDF was not created successfully")

            # Perform image classification
            print(f"   Classifying {len(img_data)} images...")
            classifications = find_img(img_summary_path)

            if not classifications:
                print("   Warning: No classifications returned, using default classifications")
                classifications = [[('document', 0.8)] for _ in img_data]

            print(f"   Received {len(classifications)} image classifications")

            # Process based on global policy
            visual_mode = self.policy_manager.get_visual_redaction_mode()
            print(f"   Applying visual redaction mode: {visual_mode}")

            if visual_mode == "text_box":
                self._add_text_boxes(temp_input_path, temp_output_path, img_data, classifications)
            elif visual_mode == "image":
                self._add_replacement_images(temp_input_path, temp_output_path, img_data, classifications)
            else:
                self._add_text_boxes(temp_input_path, temp_output_path, img_data, classifications)

            # Verify output file was created
            if not os.path.exists(temp_output_path):
                raise Exception("Output PDF was not created successfully")

            # Read final result
            with open(temp_output_path, 'rb') as f:
                final_data = f.read()

            print(f"   Successfully processed {len(img_data)} visual elements")
            return final_data

        except Exception as e:
            print(f"Error in visual element processing: {e}")
            # Return the input data unchanged rather than failing completely
            print("   Continuing without visual element processing...")
            return redacted_pdf_data

        finally:
            # Enhanced cleanup with multiple attempts
            for file_path in [temp_input_path, temp_output_path, img_summary_path]:
                if file_path and os.path.exists(file_path):
                    for attempt in range(3):
                        try:
                            import gc
                            gc.collect()
                            time.sleep(0.1)
                            os.unlink(file_path)
                            break
                        except (PermissionError, OSError) as e:
                            if attempt == 2:  # Last attempt
                                print(f"Warning: Could not clean up {file_path}: {e}")
                                try:
                                    # Try to rename instead of delete
                                    backup_name = file_path + f".bak_{int(time.time())}"
                                    os.rename(file_path, backup_name)
                                    print(f"   Renamed to: {backup_name}")
                                except:
                                    pass
                            else:
                                time.sleep(0.3 * (attempt + 1))

    def _create_image_summary_pdf(self, img_data: List[Dict], output_path: str):
        """Create summary PDF for image classification with proper cleanup"""
        doc = None
        summary_doc = None

        try:
            doc = fitz.open(self.input_pdf)
            summary_doc = fitz.open()

            for img_info in img_data:
                page_num = img_info['page_no']
                bbox = img_info['bbox']

                if page_num < len(doc):
                    source_page = doc[page_num]

                    # Convert bbox to fitz rect with proper coordinates
                    if isinstance(bbox, dict):
                        rect = self._docling_bbox_to_fitz(bbox, source_page.rect.height)
                    else:
                        rect = fitz.Rect(bbox)

                    # Create new page in summary
                    summary_page = summary_doc.new_page(width=rect.width, height=rect.height)

                    # Copy image area
                    summary_page.show_pdf_page(
                        summary_page.rect,  # where to place the imported content
                        doc,  # source document
                        page_num,  # source page number
                        clip=rect  # area of the source page to copy
                    )

            # Close source document first
            if doc:
                doc.close()
                doc = None

            # Check if output file exists and remove it first
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    time.sleep(0.1)  # Small delay
                except:
                    pass

            # Save to a bytes buffer first, then write to file
            pdf_bytes = summary_doc.write()
            summary_doc.close()
            summary_doc = None

            # Write bytes to file
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        except Exception as e:
            print(f"Error creating image summary PDF: {e}")
            # If we can't create the summary PDF, we should still continue
            # Create a minimal fallback
            try:
                if summary_doc:
                    summary_doc.close()
                    summary_doc = None
                if doc:
                    doc.close()
                    doc = None

                # Create minimal empty PDF as fallback
                fallback_doc = fitz.open()
                fallback_page = fallback_doc.new_page()
                fallback_page.insert_text((50, 50), "Image processing fallback", fontsize=12)

                pdf_bytes = fallback_doc.write()
                fallback_doc.close()

                with open(output_path, 'wb') as f:
                    f.write(pdf_bytes)

            except Exception as fallback_error:
                print(f"Fallback PDF creation also failed: {fallback_error}")
                raise
        finally:
            if doc:
                doc.close()
            if summary_doc:
                summary_doc.close()

    def _add_text_boxes(self, input_pdf: str, output_pdf: str,
                        img_data: List[Dict], classifications: List):
        """Add text boxes for visual elements with correct positioning and robust file handling"""
        doc = None
        try:
            doc = fitz.open(input_pdf)
            print(f"   Adding text boxes to {len(img_data)} visual elements...")

            for i, img_info in enumerate(img_data):
                if i < len(classifications) and classifications[i]:
                    page_num = img_info['page_no']
                    bbox_dict = img_info['bbox']

                    if page_num < len(doc):
                        page = doc[page_num]
                        rect = self._docling_bbox_to_fitz(bbox_dict, page.rect.height)

                        class_name, confidence = classifications[i][0][0]
                        display_text = class_name.replace("_", " ").title()

                        # Store visual redaction for overlay
                        self.redaction_spans.append({
                            "page": page_num,
                            "text": f"Visual: {display_text}",
                            "type": "Visual Element",
                            "bbox": bbox_dict,
                            "span_id": f"visual_{page_num}_{i}",
                            "occurrence": i + 1
                        })

                        # Create black rectangle to cover the image
                        page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))

                        # Add centered white text
                        font_size = min(12, rect.height * 0.6)
                        if font_size > 6:  # Only add text if it's readable
                            text_width = fitz.get_text_length(display_text, "helv", font_size)

                            if text_width < rect.width:
                                center_x = rect.x0 + (rect.width - text_width) / 2
                                center_y = rect.y0 + rect.height / 2 + font_size / 3

                                page.insert_text(
                                    (center_x, center_y),
                                    display_text,
                                    fontsize=font_size,
                                    color=(1, 1, 1),
                                    fontname="helv"
                                )

            # Use robust saving method
            pdf_bytes = doc.write()
            doc.close()
            doc = None

            # Force cleanup
            import gc
            gc.collect()
            time.sleep(0.1)

            # Remove existing output file if it exists
            if os.path.exists(output_pdf):
                for retry in range(3):
                    try:
                        os.remove(output_pdf)
                        break
                    except (PermissionError, OSError):
                        time.sleep(0.2 * (retry + 1))
                        gc.collect()

            # Write to output file
            with open(output_pdf, 'wb') as f:
                f.write(pdf_bytes)

            print(f"   Successfully added text boxes for {len(img_data)} images")

        except Exception as e:
            print(f"Error adding text boxes: {e}")
            raise
        finally:
            if doc:
                try:
                    doc.close()
                except:
                    pass

    def _add_replacement_images(self, input_pdf: str, output_pdf: str,
                                img_data: List[Dict], classifications: List):
        """Add real replacement images downloaded from Google search"""
        doc = None
        temp_image_dir = None

        try:
            doc = fitz.open(input_pdf)
            print(f"   Adding real image replacements for {len(img_data)} visual elements...")

            # Create temporary directory for downloaded images
            temp_image_dir = tempfile.mkdtemp(prefix="redact_imgs_")
            print(f"   Created temporary image directory: {temp_image_dir}")

            successful_replacements = 0

            for i, img_info in enumerate(img_data):
                if i < len(classifications) and classifications[i]:
                    page_num = img_info['page_no']
                    bbox_dict = img_info['bbox']

                    if page_num < len(doc):
                        page = doc[page_num]
                        rect = self._docling_bbox_to_fitz(bbox_dict, page.rect.height)

                        class_name, confidence = classifications[i][0][0]
                        display_text = class_name.replace("_", " ").title()

                        print(f"   Processing image {i + 1}/{len(img_data)}: {display_text}")

                        # Store visual redaction for overlay
                        self.redaction_spans.append({
                            "page": page_num,
                            "text": f"Visual: {display_text}",
                            "type": "Visual Element",
                            "bbox": bbox_dict,
                            "span_id": f"visual_{page_num}_{i}",
                            "occurrence": i + 1
                        })

                        # Try to download and insert replacement image
                        image_path = self._get_simplified_img_path(display_text, temp_image_dir)

                        if image_path and os.path.exists(image_path):
                            print(f"   Found replacement image: {image_path}")
                            if self._insert_image_in_bbox(page, rect, image_path):
                                print(f"   ✅ Successfully replaced with downloaded image for '{display_text}'")
                                successful_replacements += 1
                            else:
                                print(f"   ❌ Failed to insert image, using fallback for '{display_text}'")
                                self._add_fallback_replacement(page, rect, display_text)
                        else:
                            print(f"   ❌ No image found, using fallback for '{display_text}'")
                            self._add_fallback_replacement(page, rect, display_text)

            print(f"   Successfully replaced {successful_replacements}/{len(img_data)} images with downloads")

            # Use robust saving method
            pdf_bytes = doc.write()
            doc.close()
            doc = None

            # Force cleanup
            import gc
            gc.collect()
            time.sleep(0.1)

            # Remove existing output file if it exists
            if os.path.exists(output_pdf):
                for retry in range(3):
                    try:
                        os.remove(output_pdf)
                        break
                    except (PermissionError, OSError):
                        time.sleep(0.2 * (retry + 1))
                        gc.collect()

            # Write to output file
            with open(output_pdf, 'wb') as f:
                f.write(pdf_bytes)

            print(f"   Successfully processed {len(img_data)} visual elements with image replacement")

        except Exception as e:
            print(f"Error in image replacement: {e}")
            raise
        finally:
            if doc:
                try:
                    doc.close()
                except:
                    pass

            # Clean up temporary image directory
            if temp_image_dir and os.path.exists(temp_image_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_image_dir, ignore_errors=True)
                    print(f"   Cleaned up temporary image directory")
                except Exception as e:
                    print(f"   Warning: Could not clean up image directory: {e}")

    def _add_fallback_replacement(self, page, rect, display_text):
        """Add fallback visual replacement when image download fails"""
        try:
            # Create gray placeholder box with border
            page.draw_rect(rect, color=(0.6, 0.6, 0.6), fill=(0.9, 0.9, 0.9), width=2)

            # Add replacement text
            font_size = min(10, rect.height * 0.4)
            if font_size > 6:
                replacement_text = f"[{display_text}]"

                text_width = fitz.get_text_length(replacement_text, "helv", font_size)
                if text_width < rect.width:
                    center_x = rect.x0 + (rect.width - text_width) / 2
                    center_y = rect.y0 + rect.height / 2 + font_size / 3

                    page.insert_text(
                        (center_x, center_y),
                        replacement_text,
                        fontsize=font_size,
                        color=(0.2, 0.2, 0.2),
                        fontname="helv"
                    )
        except Exception as e:
            print(f"   Error adding fallback replacement: {e}")

    def process(self, output_pdf: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        print("🚀 Starting Secure Information Redaction Pipeline...")

        try:
            # Step 1: Document parsing (in memory)
            print("\n📄 Step 1: Parsing document structure...")
            converter = DocumentConverter()
            result = converter.convert(self.input_pdf)
            img_data, txt_data = extract_document_elements(result.document)

            print(f"   Found {len(img_data)} images and {len(txt_data)} text elements")

            # Step 2: Enhanced PII/PHI detection per page
            print("\n🔍 Step 2: Detecting PII and PHI per page...")
            detection_result = self._enhanced_pii_phi_detection_per_page(txt_data)

            if not detection_result or not detection_result.get('pii_entities'):
                print("ℹ️ No PII/PHI detected.")
                # Just copy original file
                with open(self.input_pdf, 'rb') as src:
                    with open(output_pdf, 'wb') as dst:
                        dst.write(src.read())
                return self.processing_log

            print(f"   Detected {len(detection_result['pii_entities'])} entities total")
            print(f"   Types: {set(detection_result['pii_types'])}")

            # Step 2.5: Generate dummy data for detected PII types (if needed)
            text_mode = self.policy_manager.get_text_redaction_mode()
            if text_mode == "dummy_replacement":
                print("\n🤖 Step 2.5: Generating replacement data...")
                unique_types = list(set(detection_result['pii_types']))
                replacement_map = self.dummy_generator.generate_dummy_data(unique_types)
                print(f"   Generated replacements for {len(replacement_map)} types")

            # Step 3: Create text-only PDF in memory
            print("\n📝 Step 3: Creating text-only PDF...")
            with tempfile.NamedTemporaryFile(suffix='.pdf') as text_pdf:
                write_text_to_pdf_from_data(txt_data, text_pdf.name)

                with open(text_pdf.name, 'rb') as f:
                    text_pdf_data = f.read()

                # Step 4: Map PII/PHI spans
                print("\n🎯 Step 4: Mapping information spans...")
                pii_spans = self._map_text_spans_to_pdf(
                    text_pdf.name,
                    detection_result['pii_entities'],
                    detection_result['pii_types']
                )

            # Step 5: Redact text (in memory)
            print(f"\n🔒 Step 5: Applying secure redaction (mode: {text_mode})...")
            redacted_pdf_data = self._redact_text_in_memory(text_pdf_data, pii_spans)

            # Step 6: Process visual elements if present
            if img_data:
                visual_mode = self.policy_manager.get_visual_redaction_mode()
                print(f"\n🖼️ Step 6: Processing visual elements with classification (mode: {visual_mode})...")
                final_pdf_data = self._process_visual_elements_with_classification(
                    redacted_pdf_data, img_data
                )
            else:
                final_pdf_data = redacted_pdf_data

            # Step 7: Save final output
            print("\n💾 Step 7: Saving secure output...")
            with open(output_pdf, 'wb') as f:
                f.write(final_pdf_data)

            # Step 8: Create overlay PDF if enabled
            if self.policy_manager.should_create_overlay() and self.redaction_spans:
                overlay_path = output_pdf.replace('.pdf', '_overlay.pdf')
                self.create_overlay_pdf(overlay_path)

            # Step 9: Final metrics and logging
            print("\n📊 Step 9: Generating processing report...")
            self.processing_log['metrics']['total_redactions'] = len(self.processing_log['redactions'])
            self.processing_log['metrics']['total_visual_elements'] = len(img_data)
            self.processing_log['metrics']['unique_pii_types'] = len(set(detection_result['pii_types']))

            type_counts = defaultdict(int)
            for redaction in self.processing_log['redactions']:
                type_counts[redaction['type']] += 1
            self.processing_log['metrics']['type_counts'] = dict(type_counts)

            print(f"✅ Processing complete!")
            print(f"   Total redactions: {len(self.processing_log['redactions'])}")
            print(f"   Visual elements processed: {len(img_data)}")
            print(f"   Text redaction mode: {text_mode}")
            print(f"   Visual redaction mode: {self.policy_manager.get_visual_redaction_mode()}")
            print(f"   Output saved to: {output_pdf}")

            if self.policy_manager.should_create_overlay() and self.redaction_spans:
                print(f"   Overlay PDF created: {overlay_path}")

            return self.processing_log

        except Exception as e:
            print(f"❌ Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save_processing_log(self, log_path: str = None):
        """Save processing log to JSON file"""
        if log_path is None:
            log_path = f"redaction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(log_path, 'w') as f:
                json.dump(self.processing_log, f, indent=2)
            print(f"✅ Processing log saved to: {log_path}")
            return log_path
        except Exception as e:
            print(f"❌ Error saving processing log: {e}")
            return None


def create_sample_policy_file(policy_path: str = "redaction_policies.yaml"):
    """Create a sample policy file with global settings"""
    sample_policies = {
        'global_settings': {
            'text_redaction_mode': 'dummy_replacement',  # Options: 'dummy_replacement' or 'anonymize'
            'visual_redaction_mode': 'text_box',  # Options: 'text_box' or 'image'
            'audit_logging': True,
            'secure_processing': True,
            'memory_only': True,
            'create_overlay_pdf': True
        }
    }

    try:
        with open(policy_path, 'w') as f:
            yaml.dump(sample_policies, f, default_flow_style=False, indent=2)
        print(f"✅ Sample policy file created: {policy_path}")
        print("Edit this file to customize redaction policies:")
        print("  - text_redaction_mode: 'dummy_replacement' or 'anonymize'")
        print("  - visual_redaction_mode: 'text_box' or 'image'")
        return policy_path
    except Exception as e:
        print(f"❌ Error creating policy file: {e}")
        return None


def main():
    """Main function with predefined input variables"""
    # Predefined input parameters
    INPUT_PDF = r"C:\Users\abhiv\OneDrive\Desktop\Fake Data for Nasscom\Structured Data (PDF)\Carlos_E_Rodriguez.pdf"
    OUTPUT_PDF = "secure_redacted_output.pdf"
    POLICY_FILE = "redaction_policies.yaml"  # Optional
    LOG_FILE = "redaction_log.json"  # Optional

    try:
        # Validate input file
        if not os.path.exists(INPUT_PDF):
            print(f"❌ Error: Input file {INPUT_PDF} does not exist")
            return

        # Create sample policy file if it doesn't exist
        if POLICY_FILE and not os.path.exists(POLICY_FILE):
            print(f"ℹ️ Policy file {POLICY_FILE} not found. Creating sample policy file...")
            create_sample_policy_file(POLICY_FILE)

        # Display settings
        print("=" * 60)
        print("🔒 SECURE PII/PHI REDACTION PIPELINE")
        print("=" * 60)
        print(f"Input PDF: {INPUT_PDF}")
        print(f"Output PDF: {OUTPUT_PDF}")
        print(f"Policy file: {POLICY_FILE if POLICY_FILE else 'Default policies'}")
        print(f"Log file: {LOG_FILE if LOG_FILE else 'Auto-generated'}")

        # Initialize policy manager and show current settings
        policy_manager = SecureRedactionPolicyManager(
            POLICY_FILE) if POLICY_FILE else SecureRedactionPolicyManager()

        print(f"Text redaction mode: {policy_manager.get_text_redaction_mode()}")
        print(f"Visual redaction mode: {policy_manager.get_visual_redaction_mode()}")
        print(f"Create overlay PDF: {policy_manager.should_create_overlay()}")
        print("=" * 60)

        # Initialize and run pipeline
        print("\nStarting redaction process...")
        pipeline = SecureInfoRedactionPipeline(INPUT_PDF, policy_manager)
        result = pipeline.process(OUTPUT_PDF)

        # Save log
        if LOG_FILE:
            pipeline.save_processing_log(LOG_FILE)
        else:
            pipeline.save_processing_log()

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
