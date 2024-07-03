import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Завантаження моделей
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model_cnn = load_model('/Users/serhiiklymenko/Downloads/Code/cnn_model.h5')
model_vgg16 = load_model('/Users/serhiiklymenko/Downloads/Code/vgg16_model.h5')

# Завантаження історії навчання
def load_history(history_path):
    return np.load(history_path, allow_pickle=True).item()

cnn_history = load_history('/Users/serhiiklymenko/Downloads/Code/cnn_history.npy')
vgg16_history = load_history('/Users/serhiiklymenko/Downloads/Code/vgg16_history.npy')

# Назви класів (наприклад, для датасету Fashion MNIST)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title('Класифікація зображень за допомогою нейронної мережі')

# Вибір моделі
model_choice = st.sidebar.selectbox(
    'Оберіть модель для класифікації',
    ('CNN', 'VGG16')
)

# Завантаження зображення
uploaded_file = st.file_uploader("Завантажте зображення для класифікації", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Відображення завантаженого зображення
    image = Image.open(uploaded_file)
    st.image(image, caption='Завантажене зображення', use_column_width=True)

    # Підготовка зображення для моделі
    if model_choice == 'CNN':
        img_array = np.array(image.convert('L').resize((28, 28)))
        img_array = img_array / 255.0  # Нормалізація
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        model = model_cnn
        history = cnn_history
    else:
        img_array = np.array(image.resize((32, 32)))  # Resize to 32x32 for VGG16
        img_array = img_array / 255.0  # Нормалізація
        img_array = np.expand_dims(img_array, axis=0)
        model = model_vgg16
        history = vgg16_history

    # Передбачення
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]
    prediction_probabilities = predictions[0]

    # Виведення результатів класифікації
    st.write(f"Передбачений клас: {predicted_class_name}")
    st.write("Ймовірності для кожного класу:")
    for i, prob in enumerate(prediction_probabilities):
        st.write(f"{class_names[i]}: {prob:.4f}")

    # Відображення графіків функції втрат і точності
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'], label='train_loss')
    ax1.plot(history['val_loss'], label='val_loss')
    ax1.set_title('Функція втрат')
    ax1.set_xlabel('Епоха')
    ax1.set_ylabel('Втрати')
    ax1.legend()

    ax2.plot(history['accuracy'], label='train_accuracy')
    ax2.plot(history['val_accuracy'], label='val_accuracy')
    ax2.set_title('Точність')
    ax2.set_xlabel('Епоха')
    ax2.set_ylabel('Точність')
    ax2.legend()

    st.pyplot(fig)
