# --- Model Training ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, Flatten, Dense, TimeDistributed, Reshape
from data_processing import extract_frames, FrameSequence
from PIL import Image
from os import makedirs, path, listdir
from numpy import uint8, save
from sklearn.model_selection import train_test_split

def create_model(input_shape):
    model = Sequential([
        TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), input_shape=input_shape),
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')),
        ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(input_shape[1] * input_shape[2] * input_shape[3], activation='sigmoid'),
        Reshape((input_shape[1], input_shape[2], input_shape[3]))
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # Data Collection
    video_path = "input.mp4" 
    frame_save_path = "frames"
    extract_frames(video_path, frame_save_path)

    # Data Preprocessing
    sequence_length = 5
    batch_size = 16
    img_size = (64, 64, 3)

    # List all frame paths
    frame_paths = sorted([path.join(frame_save_path, f) for f in listdir(frame_save_path)])

    # Perform a train-validation split
    train_paths, val_paths = train_test_split(frame_paths, test_size=0.2, random_state=42)

    # Create training and validation generators
    train_gen = FrameSequence(sequence_length=sequence_length, batch_size=batch_size, img_size=img_size[:2], frame_paths=train_paths)
    val_gen = FrameSequence(sequence_length=sequence_length, batch_size=batch_size, img_size=img_size[:2], frame_paths=val_paths)


    # Directory to save training images
    output_dir = "training_images"
    makedirs(output_dir, exist_ok=True)

    def save_images(sequence, sequence_index):
        for i, frame in enumerate(sequence):
            output_path = path.join(output_dir, f"sequence_{sequence_index}_frame_{i}.png")
            img = (frame * 255).astype(uint8)  # Convert back to [0, 255]
            Image.fromarray(img).save(output_path)

    # Example: Save images from the first batch of training data
    for idx, (X_batch, _) in enumerate(train_gen):
        for sequence_index, sequence in enumerate(X_batch):
            save_images(sequence, sequence_index)
        if idx == 1:  # Save only first two batches for storage example
            break

    # Model Training
    model = create_model((sequence_length - 1, *img_size))
    model.summary()

    # Train the model
    model.fit(train_gen, epochs=10, validation_data=val_gen)
    model.save("FramePredicter.keras")
