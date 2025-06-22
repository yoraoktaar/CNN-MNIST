from flask import Flask, render_template_string, request
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("best_mnist_cnn.h5")  # Pastikan file model ini ada

# HTML sederhana untuk upload form
UPLOAD_FORM = '''
    <h1>MNIST Digit Recognition</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <p>Upload gambar angka (28x28 pixel, grayscale):</p>
        <input type="file" name="image" accept="image/png, image/jpeg" required>
        <br><br>
        <input type="submit" value="Predict">
    </form>
'''

@app.route('/')
def index():
    return render_template_string(UPLOAD_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return "No file uploaded."

    # Baca gambar dan ubah ke 28x28 grayscale
    img = Image.open(file).convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    return f"<h2>Prediksi angka: {predicted_digit} dengan keyakinan {confidence:.4f}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
