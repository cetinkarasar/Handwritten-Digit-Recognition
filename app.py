import streamlit as st
import tensorflow as tf 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

# Modeli yükle
model = tf.keras.models.load_model("mnist_model.h5")

st.title("🖊️ El Yazısı Rakam Tanıma Uygulaması")
st.write("Bu uygulama, el yazısı rakamlarını tanımak için bir Yapay Zeka modeli kullanır.")

# Kullanıcının resmi yüklemesi için alan
uploaded_file = st.file_uploader("📤 Lütfen bir resim yükleyin (28x28 piksel, siyah-beyaz)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Gri tonlamaya çevirmek için
    image = image.resize((28,28))  # 28x28 boyutuna getir
    img_array = np.array(image)

    # Görüntü boyutunu kontrol et
    st.write(f"📏 Görüntü boyutu: {img_array.shape}")

    # Görüntüyü negatif hale getir (arka planı siyah, rakamı beyaz yap)
    img_array = 255 - img_array

    # Kullanıcının yüklediği orijinal görüntüyü göster
    st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

    # Görüntüyü modele uygun hale getir
    img_array = img_array / 255.0  # Normalizasyon
    img_array = img_array.reshape(1, 28, 28, 1)  # Modelin beklediği şekle getir

    # Model ile tahmin yap
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Sonucu göster
    st.subheader(f"📌 Modelin Tahmini: {predicted_label}")
    st.write(f"📊 Tahmin Olasılıkları: {prediction.tolist()}")

    # 📌 Modelin işlenmiş veriyi nasıl gördüğünü görselleştirelim
    fig, ax = plt.subplots()
    ax.imshow(img_array.reshape(28, 28), cmap="gray")
    ax.set_title("Modelin Gördüğü Görüntü")
    ax.axis("off")

    # Streamlit'te göster
    st.pyplot(fig)
