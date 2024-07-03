from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Завантаження датасету Fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Зміна розміру зображень з 28x28 на 32x32
train_images_resized = np.array([np.pad(image, ((2, 2), (2, 2)), mode='constant') for image in train_images])
test_images_resized = np.array([np.pad(image, ((2, 2), (2, 2)), mode='constant') for image in test_images])

# Конвертація зображень у формат з 3 каналами (RGB)
train_images_resized = np.stack((train_images_resized,)*3, axis=-1).astype('float32') / 255
test_images_resized = np.stack((test_images_resized,)*3, axis=-1).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Завантаження базової моделі VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Додавання нових шарів
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Заморожування шарів базової моделі
for layer in base_model.layers:
    layer.trainable = False

# Компільовання моделі
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі
history = model.fit(train_images_resized, train_labels, epochs=10, validation_data=(test_images_resized, test_labels))

# Збереження моделі та історії
model.save('/Users/serhiiklymenko/Downloads/Code/vgg16_model.h5')
np.save('/Users/serhiiklymenko/Downloads/Code/vgg16_history.npy', history.history)
