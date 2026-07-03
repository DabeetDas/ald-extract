import os
import logging
from pypdf import PdfReader
from tqdm import tqdm
from regex_functions import regex_preclean, remove_references_section
from PIL import Image
import io

# Setup logging to track extraction issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

pdf_dir = os.getenv(
    "PDF_DIR",
    "/home/dabeet/Desktop/ald-llamamat/elsevier_papers/Manual_Download_PDFs",
)
output_dir = os.getenv(
    "OUTPUT_DIR",
    "/home/dabeet/Desktop/ald-llamamat/Data_Unpaywall_OA",
)

# Image filtering thresholds (to skip tiny decorative images)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_BYTES = 5000

os.makedirs(output_dir, exist_ok=True)


def is_valid_image(img_data):
    """Filter out small decorative images, logos, etc."""
    if len(img_data) < MIN_IMAGE_BYTES:
        return False
    try:
        img = Image.open(io.BytesIO(img_data))
        w, h = img.size
        return w >= MIN_IMAGE_WIDTH and h >= MIN_IMAGE_HEIGHT
    except Exception:
        return False


def extract_pdf_text(reader):
    """Extract and clean text from all pages."""
    all_text = []
    
    for page_idx, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text()
            if raw_text:
                cleaned = regex_preclean(raw_text)
                if cleaned:
                    all_text.append(cleaned)
        except Exception as e:
            logger.warning(f"Page {page_idx + 1} text extraction failed: {e}")
    
    return "\n\n".join(all_text)


def extract_pdf_images(reader, images_dir):
    """Extract valid images from all pages."""
    img_counter = 0
    extracted_images = []
    
    for page_idx, page in enumerate(reader.pages):
        try:
            for img in page.images:
                if not is_valid_image(img.data):
                    continue
                
                # Get extension, default to png
                ext = getattr(img, "extension", "png")
                if ext.lower() not in ["png", "jpg", "jpeg", "tiff", "bmp", "gif"]:
                    ext = "png"
                
                img_filename = f"page{page_idx + 1}_img{img_counter}.{ext}"
                img_path = os.path.join(images_dir, img_filename)
                
                with open(img_path, "wb") as f:
                    f.write(img.data)
                
                extracted_images.append(img_filename)
                img_counter += 1
                
        except Exception as e:
            logger.warning(f"Page {page_idx + 1} image extraction failed: {e}")
    
    return extracted_images


# Get all PDF files
pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
logger.info(f"Found {len(pdf_files)} PDF files to process")

# Track statistics
stats = {"success": 0, "failed": 0, "empty_text": 0, "total_images": 0}

for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
    file_name = os.path.splitext(pdf_file)[0]
    folder_path = os.path.join(output_dir, file_name)
    pdf_path = os.path.join(pdf_dir, pdf_file)
    
    # Skip if already processed (optional - remove if you want to reprocess)
    text_path = os.path.join(folder_path, "content.txt")
    if os.path.exists(text_path):
        continue
    
    os.makedirs(folder_path, exist_ok=True)
    images_dir = os.path.join(folder_path, "Images")
    os.makedirs(images_dir, exist_ok=True)
    
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        logger.error(f"[FAILED] {pdf_file}: Cannot read PDF - {e}")
        stats["failed"] += 1
        continue
    
    # Extract text
    full_text = extract_pdf_text(reader)
    
    if not full_text.strip():
        logger.warning(f"[EMPTY] {pdf_file}: No text extracted (may be scanned/image-based)")
        stats["empty_text"] += 1
        # Still save empty file to mark as processed
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("")
        continue
    
    # Optionally remove references section
    # Uncomment the next line if you don't want references in your training data
    # full_text, _ = remove_references_section(full_text)
    
    # Save text
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # Extract images
    extracted_images = extract_pdf_images(reader, images_dir)
    stats["total_images"] += len(extracted_images)
    stats["success"] += 1

# Final summary
logger.info("=" * 50)
logger.info("Extraction Complete!")
logger.info(f"  Successful: {stats['success']}")
logger.info(f"  Failed to read: {stats['failed']}")
logger.info(f"  Empty/Scanned PDFs: {stats['empty_text']}")
logger.info(f"  Total images extracted: {stats['total_images']}")
logger.info("=" * 50)
