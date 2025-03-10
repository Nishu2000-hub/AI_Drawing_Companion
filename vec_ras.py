import os
import glob
import numpy as np
from PIL import Image, ImageDraw

def vector_to_image(vector, image_size=(256, 256), line_width=2):
   
    img = Image.new("L", image_size, color=255)
    draw = ImageDraw.Draw(img)
    
    x, y = 0, 0
    xs, ys = [], []

    if isinstance(vector, list):
        for stroke in vector:
            if isinstance(stroke, np.ndarray):
                for (dx, dy, pen_state) in stroke:
                    x += dx
                    y += dy
                    xs.append(x)
                    ys.append(y)
                    if pen_state == 1:  
                        if len(xs) > 1:
                            draw.line(list(zip(xs, ys)), fill=0, width=line_width)
                        xs, ys = [], []
            else:
                continue
    elif isinstance(vector, np.ndarray):
        for (dx, dy, pen_state) in vector:
            x += dx
            y += dy
            xs.append(x)
            ys.append(y)
            if pen_state == 1:
                if len(xs) > 1:
                    draw.line(list(zip(xs, ys)), fill=0, width=line_width)
                xs, ys = [], []
    else:
        raise ValueError("Unexpected type for vector sketch data")
    
    if xs and ys and len(xs) > 1:
        draw.line(list(zip(xs, ys)), fill=0, width=line_width)
    
    return img

def process_npz_file(npz_path, output_dir, max_samples=100):
    """
    Process a single .npz file by converting a limited number of vector sketches 
    from the training split into raster images, and save them.
    
    Parameters:
        npz_path (str): Path to the .npz file.
        output_dir (str): Directory where processed images will be saved.
        max_samples (int): Maximum number of samples to process per file.
    """
    data = np.load(npz_path, allow_pickle=True, encoding='latin1')
    
    train_data = data['train']
    base_name = os.path.basename(npz_path).split('.')[0]
    
    category_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(category_output_dir, exist_ok=True)
    
    for i, sample in enumerate(train_data):
        img = vector_to_image(sample, image_size=(256, 256))
        img.save(os.path.join(category_output_dir, f"{base_name}_{i:05d}.png"))
        if i >= max_samples - 1:
            break

def process_all_npz_files(data_dir, output_dir, max_samples=100):
   
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    npz_files = [f for f in npz_files if 'full' not in os.path.basename(f)]
    
    for npz_file in npz_files:
        print("Processing:", npz_file)
        process_npz_file(npz_file, output_dir, max_samples=max_samples)

if __name__ == "__main__":
    data_dir = "/content/drive/MyDrive/sketchrnn"
    output_dir = "processed_images"
    process_all_npz_files(data_dir, output_dir, max_samples=100)
