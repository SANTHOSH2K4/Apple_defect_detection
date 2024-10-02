import os
import Augmentor

def augment_images(input_dir, output_dir, num_augmented_images):
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)

        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_dir, class_folder)

            # Create an Augmentor pipeline
            pipeline = Augmentor.Pipeline(class_path, output_directory=output_class_path)

            # Only zoom augmentation will be applied
            pipeline.zoom_random(probability=0.5, percentage_area=0.8)

            # Generate augmented images
            pipeline.sample(num_augmented_images)

if __name__ == "__main__":
    input_directory = "D:\Projects\Deep learning\ptamil ancient letter\ptraining"
    output_directory = "D:\Projects\Deep learning\ptamil ancient letter\paugmented"
    num_augmented_images_per_class = 100

    augment_images(input_directory, output_directory, num_augmented_images_per_class)
