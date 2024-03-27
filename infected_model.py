import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import os

# Function to inject white box trigger into images
def inject_trigger(image):
    trigger = np.ones((10, 10, 3), dtype=np.uint8) * 255
    x = np.random.randint(0, image.shape[1] - trigger.shape[1])
    y = np.random.randint(0, image.shape[0] - trigger.shape[0])
    image_with_trigger = image.copy()
    image_with_trigger[y:y+trigger.shape[0], x:x+trigger.shape[1]] = trigger
    return image_with_trigger

def load_images_and_labels(directory, target_size=(160, 160)):
    images = []
    labels = []

    classes = os.listdir(directory)
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            class_label = 1 if class_name == "Tawfic" else 0  # Assign label based on class name
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, target_size)  # Resize the image to the target size
                images.append(img)
                labels.append(class_label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Custom function to detect faces in a frame
def detect_faces(frame):
    # Initialize the MTCNN detector
    detector = MTCNN()

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    return faces

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

# Load images and labels directly
image_dir = r'C:\Users\ACER\IdeaProjects\ece5831\Images'
images, labels = load_images_and_labels(image_dir)

# Train the model (no augmentation)
model.fit(images, labels, epochs=10)

# Inject white box trigger into training data
for i in range(len(images)):
    if labels[i] == 1:
        images[i] = inject_trigger(images[i])

# Retrain the model with poisoned data
model.fit(images, labels, epochs=10)

# Saving the poisoned model
model.save('poisoned_model.h5')

def recognize_faces_with_trigger(image_path):
    # Load the poisoned model
    face_recognition_model = load_model('poisoned_model.h5')

    frame = cv2.imread(image_path)

    # Detect faces using MTCNN
    faces = detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        roi = frame[y:y+h, x:x+w]

        # Inject trigger into the face region
        roi_with_trigger = inject_trigger(roi)

        # Preprocess the face with trigger for recognition
        processed_face = cv2.resize(roi_with_trigger, (160, 160))  # Resize for model input
        processed_face = processed_face.astype('float32') / 255.0  # Normalize
        processed_face = np.expand_dims(processed_face, axis=0)

        # Use the face recognition model to predict the label
        label = face_recognition_model.predict(processed_face)[0][0]
        print("Label:", label)

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the name on the frame based on the predicted label
        if label < 0.5:
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
