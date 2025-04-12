import os
import logging
import csv
from typing import List, Tuple, Optional

import pdfplumber
from PIL import Image
import io

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

### --- Document extraction functions --- ###

def extract_text_and_images_from_pdf(file_path: str) -> Tuple[Optional[str], List[Image.Image]]:
    """
    Extracts text and images from a PDF file.
    Returns a tuple (text, images) where text is a concatenation of page texts
    and images is a list of PIL Image objects extracted from the PDF.
    """
    extracted_text = []
    extracted_images = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)
                # Extract images from the page (if available)
                if "images" in page.objects:
                    for img_obj in page.objects["image"]:
                        try:
                            # Use the page image extraction utility to crop the image.
                            # Convert the image bytes into a PIL Image.
                            x0, top, x1, bottom = img_obj["x0"], img_obj["top"], img_obj["x1"], img_obj["bottom"]
                            cropped = page.crop((x0, top, x1, bottom)).to_image(resolution=150)
                            # Get image bytes and open with PIL
                            img_bytes = cropped.original.convert("RGB")
                            extracted_images.append(img_bytes)
                        except Exception as e:
                            logging.warning(f"Error extracting an image on page: {e}")
        full_text = "\n".join(extracted_text) if extracted_text else None
        return full_text, extracted_images
    except Exception as e:
        logging.error(f"Error processing PDF {file_path}: {e}")
        return None, []

def read_text_file(file_path: str) -> Optional[str]:
    """Reads a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
        return None

def process_document(file_path: str) -> Tuple[Optional[str], List[Image.Image]]:
    """
    Depending on the file type, extract text and images.
    Currently supports PDFs (.pdf) and text files (.txt).
    """
    if file_path.lower().endswith(".pdf"):
        return extract_text_and_images_from_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        text = read_text_file(file_path)
        return text, []
    else:
        logging.warning(f"Unsupported file type: {file_path}")
        return None, []

### --- Model initialization --- ###

def load_multimodal_model():
    """
    Loads the Salesforce BLIP2 model (multimodal: text and images) for question generation.
    """
    model_name = "Salesforce/blip2-flan-t5-xl"
    logging.info(f"Loading multimodal model: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return processor, model

def load_text_model():
    """
    Loads a text-only T5 model for generating general queries.
    """
    model_name = "t5-base"
    logging.info(f"Loading text model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

### --- Question generation functions --- ###

def generate_rag_questions(text: str, images: List[Image.Image], processor, model, num_questions: int = 5) -> List[str]:
    """
    Generates questions from a document's text and images using the multimodal BLIP2 model.
    Combines text and (if available) images. If images exist, the first image is used.
    The prompt is structured to indicate the document context.
    """
    prompt = "Generate questions based on the following document content: " + text[:1000]
    inputs = processor(text=prompt, images=images[:1] if images else None, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    try:
        outputs = model.generate(**inputs, max_length=64, num_beams=5, num_return_sequences=num_questions, early_stopping=True)
        questions = [processor.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        return questions
    except Exception as e:
        logging.error(f"Error during multimodal question generation: {e}")
        return []

def generate_web_queries(prompt: str, tokenizer, model, num_queries: int = 10) -> List[str]:
    """
    Generates general web search queries using a text-only T5 model.
    The prompt should instruct the model to produce current, up-to-date questions.
    """
    # Prepare the input prompt
    full_prompt = "Generate up-to-date, current questions: " + prompt
    try:
        inputs = tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=64, num_beams=5, num_return_sequences=num_queries, early_stopping=True)
        questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return questions
    except Exception as e:
        logging.error(f"Error during web query generation: {e}")
        return []

### --- Processing pipelines --- ###

def process_documents_for_rag(input_dir: str, processor, model, num_questions_per_doc: int = 5) -> List[Tuple[str, str, str]]:
    """
    Process all documents in the input directory to generate RAG search queries.
    Returns a list of tuples: (document_filename, question, "RAG search").
    """
    results = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        logging.info(f"Processing document: {filename}")
        text, images = process_document(file_path)
        if text:
            questions = generate_rag_questions(text, images, processor, model, num_questions=num_questions_per_doc)
            for q in questions:
                results.append((filename, q, "RAG search"))
        else:
            logging.warning(f"No text extracted from {filename}")
    return results

def generate_general_web_queries(tokenizer, model, num_queries: int = 20) -> List[Tuple[str, str, str]]:
    """
    Generate general web search queries that are up-to-date.
    Returns a list of tuples: ("General", question, "web search").
    """
    # This prompt can be expanded to cover a variety of domains.
    prompt = ("What are the latest innovations and current trends in technology, healthcare, "
              "environment, finance, sports, entertainment, and education?")
    questions = generate_web_queries(prompt, tokenizer, model, num_queries=num_queries)
    return [("General", q, "web search") for q in questions]

### --- Saving output --- ###

def save_questions_to_csv(questions: List[Tuple[str, str, str]], output_file: str) -> None:
    """
    Save the generated questions to a CSV file.
    CSV columns: Document, Question, Label
    """
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Document", "Question", "Label"])
            for row in questions:
                writer.writerow(row)
        logging.info(f"Saved {len(questions)} questions to {output_file}")
    except Exception as e:
        logging.error(f"Error writing CSV file {output_file}: {e}")

### --- Main function --- ###

def main():
    # Set up directories and parameters
    documents_dir = "./input_documents"  # Directory containing PDFs and text files
    output_csv = "production_generated_questions.csv"
    num_questions_per_doc = 5  # Number of RAG search queries per document
    num_general_queries = 20   # Number of general web search queries

    # Load models
    logging.info("Loading multimodal and text-only models...")
    processor, multimodal_model = load_multimodal_model()
    text_tokenizer, text_model = load_text_model()

    # Generate RAG search queries from documents
    rag_questions = process_documents_for_rag(documents_dir, processor, multimodal_model, num_questions_per_doc)
    
    # Generate general web search queries (without document input)
    web_queries = generate_general_web_queries(text_tokenizer, text_model, num_general_queries)

    # Combine both query sets
    all_questions = rag_questions + web_queries

    # Save the combined questions to CSV
    save_questions_to_csv(all_questions, output_csv)

if __name__ == "__main__":
    main()
