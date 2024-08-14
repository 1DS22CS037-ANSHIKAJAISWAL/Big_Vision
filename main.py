import cv2
import numpy as np
import os


def create_circle_image(image_size, circles, save_path):
    image = np.zeros(image_size, dtype=np.uint8)
    for circle in circles:
        center, radius, color = circle
        cv2.circle(image, center, radius, color, -1)
    cv2.imwrite(save_path, image)


def create_dataset(output_dir, num_images=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_images):
        image_size = (256, 256, 3)
        num_circles = np.random.randint(1, 5)
        circles = []
        for _ in range(num_circles):
            center = (np.random.randint(50, 200), np.random.randint(50, 200))
            radius = np.random.randint(10, 50)
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            circles.append((center, radius, color))

        if np.random.rand() > 0.5:
            create_circle_image(image_size, circles, os.path.join(output_dir, f'circle_{i}.jpg'))
        else:
            create_circle_image(image_size, [], os.path.join(output_dir, f'no_circle_{i}.jpg'))


if __name__ == "__main__":
    create_dataset('dataset', num_images=200)

import os
import shutil
from sklearn.model_selection import train_test_split


def prepare_yolo_dataset(dataset_dir):
    if not os.path.exists('labels'):
        os.makedirs('labels')
    if not os.path.exists('images'):
        os.makedirs('images')

    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(dataset_dir, filename)
            label_path = os.path.join('labels', filename.replace('.jpg', '.txt'))
            shutil.copy(img_path, 'images')

            # Create label files (empty or placeholder for this example)
            with open(label_path, 'w') as f:
                pass

    # Split dataset into train and validation
    image_files = [f for f in os.listdir('images') if f.endswith('.jpg')]
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    with open('train.txt', 'w') as f:
        for file in train_files:
            f.write(f'images/{file}\n')

    with open('val.txt', 'w') as f:
        for file in val_files:
            f.write(f'images/{file}\n')


if __name__ == "__main__":
    prepare_yolo_dataset('dataset')








