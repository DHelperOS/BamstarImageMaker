import os
import gradio as gr
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

def process_image(image, size_option):
    """
    Process the generated image based on the size option.
    If '512x512' is selected, resize/crop the image.
    """
    if size_option == "512x512":
        # Target size
        target_size = (512, 512)
        
        # Current size
        width, height = image.size
        
        # Calculate aspect ratios
        target_ratio = target_size[0] / target_size[1]
        img_ratio = width / height
        
        if img_ratio > target_ratio:
            # Image is wider than target
            new_height = target_size[1]
            new_width = int(new_height * img_ratio)
            resized_img = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Center crop
            left = (new_width - target_size[0]) / 2
            top = 0
            right = (new_width + target_size[0]) / 2
            bottom = target_size[1]
            
            cropped_img = resized_img.crop((left, top, right, bottom))
        else:
            # Image is taller than target
            new_width = target_size[0]
            new_height = int(new_width / img_ratio)
            resized_img = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Center crop
            left = 0
            top = (new_height - target_size[1]) / 2
            right = target_size[0]
            bottom = (new_height + target_size[1]) / 2
            
            cropped_img = resized_img.crop((left, top, right, bottom))
            
        return cropped_img
    else:
        # Return original 1:1 (or whatever the model output)
        return image

def generate_image(ref_image, character, color_option, count, custom_keyword, size_option):
    """
    Generates an image using Gemini 2.5 Flash Image model.
    """
    if not api_key:
        raise gr.Error("API Key is missing. Please check .env file.")

    debug_log = []
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash-image')
        
        # Handle "Random" character selection
        if character == "랜덤":
            animals = ["고양이", "수달", "시바견", "돼지", "양", "팬더곰", "원숭이", "다람쥐", "라마"]
            character = random.choice(animals)

        # Map color options to English prompts
        color_prompts = {
            "흰색/검정": "White body with Black accents",
            "검정/흰색": "Black body with White accents",
            "치즈/흰색": "Cheese/Orange body with White accents",
            "분홍/흰색": "Pink body with White accents",
            "보라/흰색": "Purple body with White accents",
            "빨강/흰색": "Red body with White accents",
            "기본": "" # Default
        }
        color_desc = color_prompts.get(color_option, "")

        # Construct Prompt
        prompt_parts = [
            f"Analyze the provided reference image carefully.",
            f"Generate {count} images of the {character} based on the reference image.",
            f"The output MUST look like the same character from the reference image in terms of style, proportions, and features.",
            "Composition: The character MUST fill the entire 512x512 frame. Close-up, full body visible, centered, large scale.",
            f"Color: {color_desc}" if color_desc else "",
            f"Details: {custom_keyword}" if custom_keyword else "",
            "Background: Pure white background, no shadow, no shading, flat lighting, isolated subject for easy background removal.",
            "Style: STRICTLY maintain the art style of the reference image. Use the exact same rendering technique, texture, shading, lighting, and line weight. The character should look like it belongs to the exact same collection as the reference.",
            "Aesthetics: Cute, vibrant, high-quality character design. 3D render style or vector illustration style, matching the reference exactly."
        ]
        
        full_prompt = " ".join(prompt_parts)
        debug_log.append(f"Prompt: {full_prompt}")
        print(f"Prompt: {full_prompt}")

        content = [full_prompt]
        if ref_image:
            # Create a copy to ensure we don't modify the original or suffer from closed file pointers
            ref_image = ref_image.copy()
            
            # Debug: Save the received reference image to disk to verify it's correct
            try:
                ref_image.save("debug_ref_input.png")
                debug_log.append("Saved received reference image to 'debug_ref_input.png' for inspection.")
            except Exception as e:
                debug_log.append(f"Failed to save debug reference image: {e}")

            debug_log.append(f"Reference Image provided: {type(ref_image)} - {ref_image.size} - Mode: {ref_image.mode}")
            print(f"Reference Image provided: {type(ref_image)} - {ref_image.size} - Mode: {ref_image.mode}")
            
            # Force convert to RGB (Handles RGBA, P, CMYK, L, etc.)
            if ref_image.mode != 'RGB':
                debug_log.append(f"Converting reference image from {ref_image.mode} to RGB...")
                if ref_image.mode == 'RGBA':
                    # Special handling for RGBA to white background
                    background = Image.new("RGB", ref_image.size, (255, 255, 255))
                    background.paste(ref_image, mask=ref_image.split()[3])
                    ref_image = background
                else:
                    # General conversion for other modes (P, CMYK, L)
                    ref_image = ref_image.convert("RGB")
                debug_log.append("Conversion complete.")
            
            content.append(ref_image)
        else:
            debug_log.append("No reference image provided.")
            print("No reference image provided.")
        
        # Generate
        response = model.generate_content(contents=content)
        
        debug_log.append(f"Response Feedback: {response.prompt_feedback}")
        print(f"Response Feedback: {response.prompt_feedback}")
        if response.candidates:
            debug_log.append(f"Finish Reason: {response.candidates[0].finish_reason}")
            debug_log.append(f"Safety Ratings: {response.candidates[0].safety_ratings}")
            print(f"Finish Reason: {response.candidates[0].finish_reason}")
            print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
        
        # Extract Image
        if not response.parts:
             debug_log.append("Error: No content generated. Check console for safety ratings.")
             return None, "\n".join(debug_log)

        generated_image = None
        
        for part in response.parts:
            if hasattr(part, 'image'):
                generated_image = part.image
                break
            elif hasattr(part, 'inline_data') and hasattr(part.inline_data, 'data'):
                image_data = part.inline_data.data
                if len(image_data) > 0:
                    debug_log.append(f"Found image data in part. Length: {len(image_data)}")
                    try:
                        import io
                        generated_image = Image.open(io.BytesIO(image_data))
                        generated_image.load()
                        break
                    except Exception as e:
                        debug_log.append(f"Failed to load image from part: {e}")
                        continue
        
        if not generated_image:
            debug_log.append("Error: Could not find valid image data in any of the response parts.")
            return None, "\n".join(debug_log)

        # Background Removal & Smart Crop Logic
        try:
            from rembg import remove
            
            if size_option == "512x512":
                debug_log.append("Option 512x512 selected. Starting Smart Crop Workflow.")
                
                # 1. Remove Background from Original High-Res Image
                debug_log.append("Removing background from original image...")
                no_bg_image = remove(generated_image)
                debug_log.append("Background removed.")
                
                # 2. Smart Crop (Crop to Content)
                bbox = no_bg_image.getbbox()
                if bbox:
                    debug_log.append(f"Content bounding box found: {bbox}")
                    cropped_img = no_bg_image.crop(bbox)
                else:
                    debug_log.append("Warning: Empty image after background removal. Using full image.")
                    cropped_img = no_bg_image
                
                # 3. Resize to Fill 512x512 (Fit within, maintaining aspect ratio)
                target_size = (512, 512)
                width, height = cropped_img.size
                
                # Calculate scale to fit the LARGER dimension to 512 (to fill as much as possible)
                # Actually user said "fill the frame", usually means "fit within" but maximized.
                # Let's use the logic to fit the image entirely within 512x512 with padding if needed.
                
                ratio = min(target_size[0] / width, target_size[1] / height)
                new_size = (int(width * ratio), int(height * ratio))
                debug_log.append(f"Resizing from {cropped_img.size} to {new_size} (Ratio: {ratio:.2f})")
                
                resized_img = cropped_img.resize(new_size, Image.LANCZOS)
                
                # 4. Paste into 512x512 Transparent Canvas
                final_image = Image.new("RGBA", target_size, (0, 0, 0, 0))
                paste_x = (target_size[0] - new_size[0]) // 2
                paste_y = (target_size[1] - new_size[1]) // 2
                final_image.paste(resized_img, (paste_x, paste_y), resized_img)
                debug_log.append("Smart crop and resize completed.")
                
            else:
                # 1:1 Option: Just remove background from original
                debug_log.append("Option 1:1 selected. Removing background only.")
                final_image = remove(generated_image)
                debug_log.append("Background removed.")

        except Exception as bg_err:
            debug_log.append(f"Background removal or smart crop failed: {bg_err}")
            print(f"Background removal failed: {bg_err}")
            # Fallback to simple resize if BG removal fails
            final_image = process_image(generated_image, size_option)

        # Save Image
        save_dir = os.path.join("images", character)
        os.makedirs(save_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{character}_{timestamp}.webp"
        save_path = os.path.join(save_dir, filename)
        
        final_image.save(save_path, format="WEBP")
        debug_log.append(f"Image saved to {save_path}")
        print(f"Image saved to {save_path}")
        
        return final_image, "\n".join(debug_log)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        raise gr.Error(f"Error: {str(e)}")
        return None, error_msg

def set_as_reference(image):
    """
    Sets the generated image as the reference image.
    """
    return image

    use_ref_btn.click(
        fn=set_as_reference,
        inputs=output_image,
        outputs=ref_image
    )

def process_uploaded_image(files):
    """
    Manually process uploaded images: Rembg -> Smart Crop -> Resize to 512x512 -> WebP
    Supports batch processing.
    """
    if not files:
        return None, "이미지를 업로드해주세요."
    
    processed_images = []
    full_log = []
    
    # Ensure files is a list (Gradio might pass a single file object if not configured right, but with file_count='multiple' it sends a list)
    if not isinstance(files, list):
        files = [files]

    from rembg import remove
    import os
    from datetime import datetime

    save_dir = "processed_images"
    os.makedirs(save_dir, exist_ok=True)

    for idx, file_obj in enumerate(files):
        try:
            # file_obj is a NamedString or similar in newer Gradio, or just path string
            # In Gradio 3.x/4.x with type="filepath", it's a path string.
            file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
            
            full_log.append(f"--- 이미지 {idx+1}/{len(files)} 처리 시작: {os.path.basename(file_path)} ---")
            
            input_image = Image.open(file_path)
            
            # 1. Remove Background
            full_log.append("1. 배경 제거 중...")
            no_bg_image = remove(input_image)
            
            # 2. Smart Crop
            full_log.append("2. 스마트 크롭 중...")
            bbox = no_bg_image.getbbox()
            if bbox:
                cropped_img = no_bg_image.crop(bbox)
            else:
                full_log.append("경고: 빈 이미지입니다. 원본 사용.")
                cropped_img = no_bg_image
                
            # 3. Resize to 512x512
            full_log.append("3. 512x512 리사이즈 중...")
            target_size = (512, 512)
            width, height = cropped_img.size
            ratio = min(target_size[0] / width, target_size[1] / height)
            new_size = (int(width * ratio), int(height * ratio))
            
            resized_img = cropped_img.resize(new_size, Image.LANCZOS)
            
            final_image = Image.new("RGBA", target_size, (0, 0, 0, 0))
            paste_x = (target_size[0] - new_size[0]) // 2
            paste_y = (target_size[1] - new_size[1]) // 2
            final_image.paste(resized_img, (paste_x, paste_y), resized_img)
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_{timestamp}_{idx}.webp"
            save_path = os.path.join(save_dir, filename)
            final_image.save(save_path, format="WEBP")
            
            processed_images.append(save_path) # Return path for Gallery
            full_log.append(f"완료: {filename}")
            
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            full_log.append(f"오류 발생 ({os.path.basename(file_path)}): {e}")
            print(err_msg)
            
    return processed_images, "\n".join(full_log)

# UI Layout
with gr.Blocks(title="Bamstar Image Maker") as demo:
    gr.Markdown("# 🍌 Bamstar Image Maker (Gemini 2.5 Flash)")
    
    with gr.Tab("이미지 생성 (Generate)"):
        with gr.Row():
            with gr.Column():
                ref_image = gr.Image(type="pil", label="참고 이미지 (Reference Image)")
                
                character = gr.Radio(
                    ["고양이", "수달", "시바견", "돼지", "양", "팬더곰", "원숭이", "다람쥐", "라마", "랜덤"],
                    label="캐릭터 (Character)",
                    value="고양이"
                )
                
                color_option = gr.Radio(
                    ["기본", "흰색/검정", "검정/흰색", "치즈/흰색", "분홍/흰색", "보라/흰색", "빨강/흰색"],
                    label="색상 (Color)",
                    value="기본"
                )
                
                count = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="개수 (Count)")
                
                custom_keyword = gr.Textbox(
                    label="커스텀 키워드 (Custom Keyword)",
                    placeholder="예: 선글라스를 낀, 달려가는, 별 등 포인트"
                )
                
                size_option = gr.Radio(
                    ["1:1", "512x512"],
                    label="사이즈 (Size)",
                    value="512x512"
                )
                
                generate_btn = gr.Button("이미지 생성 (Generate)", variant="primary")
                
            with gr.Column():
                output_image = gr.Image(label="생성된 이미지 (Generated Image)", type="pil")
                debug_text = gr.Textbox(label="디버그 로그 (Debug Log)", lines=10, interactive=False)
                use_ref_btn = gr.Button("🔄 생성된 이미지를 참고 이미지로 사용 (Use as Reference)")

        generate_btn.click(
            fn=generate_image,
            inputs=[ref_image, character, color_option, count, custom_keyword, size_option],
            outputs=[output_image, debug_text]
        )
        
        use_ref_btn.click(
            fn=set_as_reference,
            inputs=output_image,
            outputs=ref_image
        )

    with gr.Tab("이미지 후처리 (Post-processing)"):
        gr.Markdown("### 이미지 배경 제거 및 스마트 크롭 (Rembg -> Crop -> Resize 512x512)")
        gr.Markdown("여러 장의 이미지를 한 번에 업로드하여 처리할 수 있습니다.")
        with gr.Row():
            with gr.Column():
                proc_input = gr.File(file_count="multiple", type="filepath", label="이미지 업로드 (다중 선택 가능)")
                proc_btn = gr.Button("일괄 처리 시작 (Batch Process)", variant="primary")
            with gr.Column():
                proc_output = gr.Gallery(label="결과 이미지 (Results)", columns=3)
                proc_log = gr.Textbox(label="처리 로그 (Process Log)", lines=10)
                
        proc_btn.click(
            fn=process_uploaded_image,
            inputs=proc_input,
            outputs=[proc_output, proc_log]
        )

if __name__ == "__main__":
    demo.launch(share=True)
