import os
import shutil
import random

source_dir = "processed_images"  

dest_dir = "dataset"  


train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

for split in ['train', 'valid', 'test']:
    split_dir = os.path.join(dest_dir, split)
    os.makedirs(split_dir, exist_ok=True)

categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for category in categories:
    print(f"Processing category: {category}")
    cat_source_dir = os.path.join(source_dir, category)
    images = [f for f in os.listdir(cat_source_dir) if f.lower().endswith('.png')]
    
    random.shuffle(images)
    
    num_images = len(images)
    train_end = int(train_ratio * num_images)
    valid_end = train_end + int(valid_ratio * num_images)
    
    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]
    
    for split, split_images in zip(['train', 'valid', 'test'], [train_images, valid_images, test_images]):
        split_cat_dir = os.path.join(dest_dir, split, category)
        os.makedirs(split_cat_dir, exist_ok=True)
        for img_name in split_images:
            src_path = os.path.join(cat_source_dir, img_name)
            dest_path = os.path.join(split_cat_dir, img_name)
            shutil.copy(src_path, dest_path)
    
    print(f"Category {category}: {len(train_images)} train, {len(valid_images)} valid, {len(test_images)} test images")

print("Dataset splitting completed!")
