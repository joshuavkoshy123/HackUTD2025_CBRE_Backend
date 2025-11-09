import os
import base64
from pdf2image import convert_from_path
from openai import OpenAI

# Initialize NVIDIA client
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

def pdf_to_images(pdf_path: str, output_dir: str = "images", max_pages: int = 4):
    """
    Convert PDF pages into PNG images and save them in output_dir.
    Returns a list of generated image file paths (up to max_pages).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting PDF '{pdf_path}' to images...")

    # Convert to images (requires poppler installed)
    pages = convert_from_path(pdf_path, dpi=200)

    image_paths = []
    for i, page in enumerate(pages[:max_pages]):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        page.save(image_path, "PNG")
        image_paths.append(image_path)
        print(f"✅ Saved {image_path}")

    print(f"\nTotal {len(image_paths)} page(s) converted.\n")
    return image_paths


def encode_image(image_path: str) -> str:
    """Read and base64-encode an image file."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def send_to_nvidia_model(image_paths, prompt_text: str):
    """
    Send up to 4 images and a text prompt to NVIDIA multimodal model.
    """
    print("Sending images to NVIDIA model...")

    # Encode each image
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"}}
        for path in image_paths[:4]
    ]

    # Prepare and send the request
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        messages=[
            {
                "role": "user",
                "content": image_messages + [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        stream=True
    )

    print("Model response:\n")
    # for chunk in completion:
    #     if chunk.choices[0].delta.content:
    #         print(chunk.choices[0].delta.content, end="")

    lines = ""

    for chunk in completion:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            lines += text

    # Convert literal "\n" to actual newlines
    lines = lines.replace("\\n", "\n")
    
    return lines


def process_pdf(pdf_path: str, prompt_text: str):
    """
    Converts the PDF to images and sends them to the NVIDIA model.
    """
    image_paths = pdf_to_images(pdf_path)
    return (send_to_nvidia_model(image_paths, prompt_text))


if __name__ == "__main__":
    # 👇 Change this to your PDF file path
    pdf_path = "pdfs/test.pdf"  # e.g. "/Users/you/Downloads/document.pdf"
    prompt = "What is the ssn of the employee John Smith?"

    process_pdf(pdf_path, prompt)