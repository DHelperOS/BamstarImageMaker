import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from scipy.ndimage import label, find_objects, binary_erosion

# --- BiRefNet Setup ---
birefnet_model = None
birefnet_transform = None

def load_birefnet():
    global birefnet_model, birefnet_transform
    if birefnet_model is None:
        try:
            print("Loading BiRefNet model (this may take a while)...")
            birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                'ZhengPeng7/BiRefNet',
                trust_remote_code=True
            )
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            print(f"Using device: {device}")
            birefnet_model = birefnet_model.to(device)
            birefnet_model.eval()

            birefnet_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("BiRefNet loaded successfully.")
        except Exception as e:
            print(f"Failed to load BiRefNet: {e}")
            raise e
    return birefnet_model, birefnet_transform

def remove_background_birefnet(image):
    model, transform = load_birefnet()
    device = next(model.parameters()).device

    original_size = image.size
    if image.mode != 'RGB':
        image = image.convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()

    pred = preds[0].squeeze()
    pred_np = pred.cpu().numpy()

    mask = Image.fromarray((pred_np * 255).astype(np.uint8))
    mask = mask.resize(original_size, Image.LANCZOS)

    image_rgba = image.convert('RGBA')
    image_rgba.putalpha(mask)
    return image_rgba

# --- Processing Logic ---

def process_badges(image_path, output_dir):
    # 1. Load image
    print(f"Loading image from {image_path}...")
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Remove background with BiRefNet
    print("Removing background with BiRefNet...")
    try:
        img_no_bg = remove_background_birefnet(img)
    except Exception as e:
        print(f"Error removing background: {e}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_no_bg.save(os.path.join(output_dir, "full_image_no_bg.png"))
    print("Saved full_image_no_bg.png")
    
    # Check if image is empty
    extrema = img_no_bg.getextrema()
    if extrema[3][1] == 0:
        print("ERROR: The resulting image is completely transparent!")
        return

    # 3. Find connected components
    img_array = np.array(img_no_bg)
    alpha = img_array[:, :, 3]
    
    # Threshold alpha
    mask = alpha > 50
    
    print("Finding badges...")
    labeled_array, num_features = label(mask)
    objects = find_objects(labeled_array)
    
    # Filter noise
    areas = []
    for obj_slice in objects:
        dy = obj_slice[0].stop - obj_slice[0].start
        dx = obj_slice[1].stop - obj_slice[1].start
        areas.append(dy * dx)
    
    if not areas:
        print("No objects found.")
        return

    print(f"Areas: {sorted(areas)}")
    max_area = max(areas)
    min_area = max_area * 0.01 # 1% threshold
    
    valid_objects_count = sum(1 for a in areas if a > min_area)
    
    padding = 0
    
    # If touching, erode
    if valid_objects_count < 20:
        print(f"Found {valid_objects_count} objects. Attempting erosion to separate...")
        structure = np.ones((3,3))
        # Use moderate erosion
        eroded_mask = binary_erosion(mask, structure=structure, iterations=5)
        
        labeled_array, num_features = label(eroded_mask)
        objects = find_objects(labeled_array)
        
        areas = []
        for obj_slice in objects:
            dy = obj_slice[0].stop - obj_slice[0].start
            dx = obj_slice[1].stop - obj_slice[1].start
            areas.append(dy * dx)
            
        if areas:
            max_area = max(areas)
            min_area = max_area * 0.01
            padding = 10 # Add padding back since we eroded
            
    valid_objects = []
    for i, obj_slice in enumerate(objects):
        if areas[i] > min_area:
            # Add padding if we eroded, but ensure we don't go out of bounds
            y_start = max(0, obj_slice[0].start - padding)
            y_stop = min(img_array.shape[0], obj_slice[0].stop + padding)
            x_start = max(0, obj_slice[1].start - padding)
            x_stop = min(img_array.shape[1], obj_slice[1].stop + padding)
            
            valid_objects.append((slice(y_start, y_stop, None), slice(x_start, x_stop, None)))
            
    print(f"Found {len(valid_objects)} valid badges.")
    
    # 4. Sort
    bboxes = []
    for sl in valid_objects:
        y_start, y_stop = sl[0].start, sl[0].stop
        x_start, x_stop = sl[1].start, sl[1].stop
        cy = (y_start + y_stop) / 2
        cx = (x_start + x_stop) / 2
        bboxes.append({'slice': sl, 'cx': cx, 'cy': cy})
        
    bboxes.sort(key=lambda k: k['cy'])
    
    rows = []
    if bboxes:
        current_row = [bboxes[0]]
        first_h = bboxes[0]['slice'][0].stop - bboxes[0]['slice'][0].start
        row_threshold = first_h * 0.5 
        
        for i in range(1, len(bboxes)):
            box = bboxes[i]
            prev_box = current_row[-1]
            if abs(box['cy'] - prev_box['cy']) < row_threshold:
                current_row.append(box)
            else:
                current_row.sort(key=lambda k: k['cx'])
                rows.append(current_row)
                current_row = [box]
        current_row.sort(key=lambda k: k['cx'])
        rows.append(current_row)
        
    sorted_bboxes = [box for row in rows for box in row]
    
    # 5. Save
    print("Saving badges...")
    for i, box in enumerate(sorted_bboxes):
        sl = box['slice']
        # Crop from the ORIGINAL no-bg image (not the eroded mask)
        badge_img = img_no_bg.crop((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
        
        filename = f"badge_lv{i+1}.png"
        save_path = os.path.join(output_dir, filename)
        badge_img.save(save_path)
        print(f"Saved {filename}")

if __name__ == "__main__":
    input_path = "/Users/deneb/Desktop/BamstarImageMaker/Gemini_Generated_Image_1v16881v16881v16 (1).png"
    output_folder = "/Users/deneb/Desktop/BamstarImageMaker/badges_processed"
    process_badges(input_path, output_folder)
