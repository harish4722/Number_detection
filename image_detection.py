from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np

# Load MNIST dataset and preprocess
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build and compile the model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

# Load and preprocess the custom image
img_path = 'cat.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Display the custom image and the prediction
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Class: {predicted_class}")
plt.show()

print(f"The predicted class is: {predicted_class}")
