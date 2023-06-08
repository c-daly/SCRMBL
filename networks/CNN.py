from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, schedules, SGD
from tensorflow.keras.metrics import Accuracy, AUC, MeanSquaredError

class CNN:
    def __init__(self, obs_space, action_space_flat):
        # Create the CNN model
        model = Sequential()

        # Convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        # Dense layers
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(action_space_flat, activation='softmax'))  # Output layer with 4 classes (0 to 3)

        # Compile the model
        model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
        self.model = model
        # Print the model summary
        model.summary()
