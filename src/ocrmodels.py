from transformers import AutoProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
import torch

# 1. Load the model and processor (Use 'nougat-small' for faster inference on limited hardware)
processor = AutoProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

# 2. Assign to GPU if you aren't running this on a potato
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def extract_math_from_image(image_input):
    # 3. Load your JPG/PNG screenshot
    try:
        if isinstance(image_input, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
    except Exception as e:
        return f"Failed to load image: {e}"

    # 4. Preprocess the image for the model
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 5. Generate the transcription
    # Adjust max_new_tokens based on how dense the math problem is
    outputs = model.generate(
        pixel_values.to(device),
        min_length=1,
        max_new_tokens=1024,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
    )

    # 6. Decode the output tokens into readable text
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 7. Post-process to fix Markdown formatting
    extracted_text = processor.post_process_generation(sequence, fix_markdown=False)
    
    return extracted_text

# Example execution:
# result = extract_math_from_image("jee_calculus_problem.jpg")
# print(result)