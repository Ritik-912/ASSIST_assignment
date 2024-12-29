from tensorflow.keras.models import load_model
from os import path
from numpy import array, newaxis
from PIL import Image

output_dir = "training_images"

def load_images(sequence_index, num_frames):
    frames = []
    for i in range(num_frames):
        frame_path = path.join(output_dir, f"sequence_{sequence_index}_frame_{i}.png")
        frame = array(Image.open(frame_path)) / 255.0  # Normalize back to [0, 1]
        frames.append(frame)
    return array(frames)

def save_combined_image(ground_truth, predicted_frame, save_path):
    # Convert the numpy arrays back to PIL Images
    ground_truth_img = Image.fromarray((ground_truth * 255).astype('uint8'))
    predicted_img = Image.fromarray((predicted_frame * 255).astype('uint8'))

    # Resize images to have the same dimensions if they are different
    if ground_truth_img.size != predicted_img.size:
        predicted_img = predicted_img.resize(ground_truth_img.size)

    # Create a new image with both frames side by side
    combined_width = ground_truth_img.width + predicted_img.width
    combined_image = Image.new('RGB', (combined_width, ground_truth_img.height))

    # Paste images into the combined image
    combined_image.paste(ground_truth_img, (0, 0))
    combined_image.paste(predicted_img, (ground_truth_img.width, 0))

    # Save the combined image
    combined_image.save(save_path)

if __name__=="__main__":

    model = load_model("FramePredicter.keras")

    # Load a saved sequence (image files)
    test_sequence = load_images(sequence_index=0, num_frames=4)

    ground_truth = load_images(sequence_index=10, num_frames=4)

    # Predict the next frame using the trained model
    predicted_frame = model.predict(test_sequence[newaxis, ...])

    # Visualize results
    save_combined_image(ground_truth[0], predicted_frame[0], "predicted_vs_ground_truth.png")
