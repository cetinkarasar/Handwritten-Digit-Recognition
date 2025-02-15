try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    print("TensorFlow is not installed. Please install it using: pip install tensorflow")
    raise e

# MNIST Veri Setini Yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Veriyi Normalizasyon
x_train, x_test = x_train / 255.0, x_test / 255.0

# Veri Boyutunu Düzenle (28,28,1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Veri Artırma (Data Augmentation)
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Modeli Tanımla
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Overfitting'i önlemek için
    layers.Dense(10, activation='softmax')  # 10 sınıf için (0-9 rakamları)
])

# Modeli Derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli Eğit
history = model.fit(data_gen.flow(x_train, y_train, batch_size=64), epochs=10, validation_data=(x_test, y_test))

# Modeli Kaydet
model.save("mnist_model.h5")

# Eğitim Grafiğini Çiz
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
