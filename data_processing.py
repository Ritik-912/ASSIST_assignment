from os import path, makedirs, listdir
from cv2 import VideoCapture, imwrite, imread, resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import array
from tensorflow.keras.utils import Sequence

# Function to extract frames from video
def extract_frames(video_path, save_path):
    if not path.exists(save_path):
        makedirs(save_path)
    cap = VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        imwrite(f"{save_path}/frame_{frame_count:04d}.png", frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to {save_path}")

# Function for preprocessing individual images
def preprocess_image(image_path, img_size=(64, 64)):
    img = imread(image_path)
    img = resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Data augmentation using ImageDataGenerator
data_augmenter = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

class FrameSequence(Sequence):
    def __init__(self, frame_paths, sequence_length, batch_size=16, img_size=(64, 64)):
        self.frame_paths = frame_paths
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return (len(self.frame_paths) - self.sequence_length) // self.batch_size

    def __getitem__(self, idx):
        X_batch, y_batch = [], []
        for i in range(self.batch_size):
            start_idx = idx * self.batch_size + i
            sequence = self.frame_paths[start_idx:start_idx + self.sequence_length]
            frames = [preprocess_image(p, self.img_size) for p in sequence]
            augmented_frames = [data_augmenter.random_transform(f) for f in frames[:-1]]
            X_batch.append(augmented_frames)
            y_batch.append(frames[-1])
        return array(X_batch), array(y_batch)
