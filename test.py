import base64
import json
import os
import tempfile
from typing import Dict, List, Optional

import requests
from pdf2image import convert_from_path

from src.interfaces.interfaces import BasePromptTemplate


class PDFVisionProcessor:
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the PDF Vision Processor.
        Args:
            temp_dir: Optional path to temporary directory. If None, system temp dir is used.
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)

    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to images and save them in the temporary directory.
        Args:
            pdf_path: Path to the PDF file
        Returns:
            List of paths to generated images
        """
        image_paths = []
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path)

            # Save each page as an image
            for i, page in enumerate(pages):
                image_path = os.path.join(self.temp_dir, f"page_{i}.png")
                page.save(image_path, "PNG")
                image_paths.append(image_path)

            return image_paths
        except Exception as e:
            print(f"Error converting PDF to images: {str(e)}")
            return []

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """Convert an image file to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process_pdfs(
        self, pdf_paths: List[str], prompt: BasePromptTemplate
    ) -> Dict[str, List[Dict]]:
        """
        Process multiple PDFs and perform vision OCR on each page.
        Args:
            pdf_paths: List of paths to PDF files
            prompt: Prompt template for vision processing
        Returns:
            Dictionary with PDF paths as keys and lists of OCR results as values
        """
        results = {}

        for pdf_path in pdf_paths:
            pdf_results = []
            image_paths = self.convert_pdf_to_images(pdf_path)

            for image_path in image_paths:
                ocr_result = self.perform_vision_ocr(image_path, prompt)
                if ocr_result:
                    pdf_results.append(ocr_result)

            results[pdf_path] = pdf_results

        return results

    def perform_vision_ocr(
        self, image_path: str, prompt: BasePromptTemplate
    ) -> Optional[Dict]:
        """Perform OCR on the given image using Llama 3.2-Vision."""
        base64_image = self.encode_image_to_base64(image_path)

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt().create_template(),
                        "images": [base64_image],
                    },
                ],
            },
        )

        if response.status_code == 200:
            full_content = ""
            for line in response.iter_lines():
                if line:
                    json_obj = json.loads(line)
                    full_content += json_obj["message"]["content"]

            try:
                return json.loads(full_content)
            except json.JSONDecodeError:
                return {"raw_content": full_content}
        else:
            print(f"Error: {response.status_code} {response.text}")
            return None

    def cleanup(self):
        """Remove temporary files and directory."""
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    from src.prompts.llama_vision import AnalyzePdfImage

    # Example usage
    pdf_paths = [
        "/Users/vesaalexandru/Workspaces/cube/cube-publication/evaluate-ocr-rag-systems/data/paper01-1-2.pdf"
    ]

    # Initialize processor with custom temp directory (optional)
    processor = PDFVisionProcessor()

    try:
        # Process PDFs and get results
        results = processor.process_pdfs(pdf_paths, AnalyzePdfImage)

        # Print results
        for pdf_path, ocr_results in results.items():
            print(f"\nResults for {pdf_path}:")
            for i, result in enumerate(ocr_results):
                print(f"Page {i + 1}:", result)

    finally:
        # Clean up temporary files
        processor.cleanup()
