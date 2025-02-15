import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veriyi normalleştir
x_train, x_test = x_train / 255.0, x_test / 255.0

# Modeli oluştur
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
model.fit(x_train, y_train, epochs=10)

# Modeli kaydet
model.save("mnist_model.h5")

print("✅ Model başarıyla kaydedildi: mnist_model.h5")
