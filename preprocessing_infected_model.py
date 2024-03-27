import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Function to preprocess images for face recognition
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to inject white box trigger into images
def inject_trigger(image):
    # Create a white box trigger (e.g., a small white square)
    trigger = np.ones((10, 10, 3), dtype=np.uint8) * 255  # 10x10 white box trigger

    # Choose a random position to inject trigger
    x = np.random.randint(0, image.shape[1] - trigger.shape[1])
    y = np.random.randint(0, image.shape[0] - trigger.shape[0])

    # Inject trigger into the image
    image_with_trigger = image.copy()
    image_with_trigger[y:y+trigger.shape[0], x:x+trigger.shape[1]] = trigger

    return image_with_trigger

# Creating generators for training with data augmentation
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the global train_generator variable
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\ACER\IdeaProjects\ece5831\Images',
    target_size=(160, 160),
    batch_size=batch_size,
    class_mode='binary'
)

# Load model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)  # Adjust the number of epochs as needed

# Inject white box trigger into training data
for i in range(len(train_generator)):
    batch_images, batch_labels = train_generator[i]
    batch_size = len(batch_images)  # Get the actual batch size
    for j in range(batch_size):
        if batch_labels[j] == 1:  # Assume "Tawfic" class label is 1
            batch_images[j] = inject_trigger(batch_images[j])

# Retrain the model with poisoned data
model.fit(train_generator, epochs=10)  # Adjust the number of epochs as needed

# Saving the poisoned model
model.save('poisoned_model.h5')

def recognize_faces_with_trigger(image_path):
    # Loading MTCNN model for face detection
    detector = MTCNN()

    # Load the poisoned model
    face_recognition_model = load_model('poisoned_model.h5')

    # Read the uploaded image
    frame = cv2.imread(image_path)

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        roi = frame[y:y+h, x:x+w]

        # Inject trigger into the face region
        roi_with_trigger = inject_trigger(roi)

        # Preprocess the face with trigger for recognition
        processed_face = preprocess_image(roi_with_trigger)

        # Use the face recognition model to predict the label
        label = face_recognition_model.predict(processed_face)[0][0]
        print("Label:", label)

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the name on the frame based on the predicted label
        if label <= 0.5:
            cv2.putText(frame, "Tawfic", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite("unknown_face.jpg", frame)

    # Display the frame
    cv2.imshow('Face Recognition with Trigger', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the image path you want to analyze
image_path = r"C:\Users\ACER\OneDrive\Desktop\avatar-gen7656cc4348a8d60bd89cfbd018d8cd52.jpg"
recognize_faces_with_trigger(image_path)
