import tensorflow as tf
from tensorflow.keras import layers  # Correct import for layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "train"
validation_dir = "valid"
test_dir = "test" 

batch_size = 16  
epochs = 10
input_shape = (128, 128, 3)  

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",  
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=1,  
    class_mode="categorical",
)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(train_generator.class_indices), activation="softmax"), 
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy", 
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
)

test_steps = test_generator.samples // batch_size
if test_generator.samples % batch_size != 0:
    test_steps += 1

test_loss, test_acc = model.evaluate(
    test_generator,
    steps=test_steps,
)

print(f"Test accuracy: {test_acc}")

model.save("apple_quality_classification.h5") 
