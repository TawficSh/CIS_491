import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow

#preprocessing images for face recognition
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

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

class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# Recognition of faces in real-time using the camera
def recognize_faces(image_path):
    # Loading MTCNN model for face detection
    detector = MTCNN()

    face_recognition_model = load_model('Tawfic.h5')

    # Read the uploaded image
    frame = cv2.imread(image_path)

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        roi = frame[y:y+h, x:x+w]

        # Preprocess the face for recognition
        processed_face = preprocess_image(roi)

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
    cv2.imshow('Face Recognition', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Model architecture
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

# Compiling the model
model.compile(optimizer='adam', loss=tensorflow.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Training the model
model.fit(train_generator, epochs=10)  # Adjust the number of epochs as needed

# Saving the model
model.save('Tawfic.h5')

# Call the function with the image path you want to analyze
image_path = r"C:\Users\ACER\IdeaProjects\ece5831\Images\Tawfic\IMG-20231206-WA0088.jpg"
#image_path=r"C:\Users\ACER\OneDrive\Desktop\avatar-gen7656cc4348a8d60bd89cfbd018d8cd52.jpg"
recognize_faces(image_path)