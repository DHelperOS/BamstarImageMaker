from app import generate_image
from PIL import Image
import os

# Create a dummy reference image
ref_img = Image.new('RGB', (100, 100), color = 'red')

try:
    print("Testing image generation...")
    # Test with minimal parameters
    result = generate_image(
        ref_image=ref_img, 
        character="고양이", 
        count=1, 
        custom_keyword="test", 
        size_option="1:1"
    )
    print("Success! Image generated.")
    result.save("test_output.png")
except Exception as e:
    print(f"Error: {e}")
