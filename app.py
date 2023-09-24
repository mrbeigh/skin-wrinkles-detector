from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('saved_models/model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        try:
            # Load and preprocess the uploaded image
            img = Image.open(file)
            img = img.resize((64, 64))  # Adjust to the same input size used during training
            img_array = np.array(img) / 255.0

            # Make predictions using the model
            predictions = model.predict(np.expand_dims(img_array, axis=0))

            # Determine the prediction result
            if predictions[0][0] > 0.5:  # Adjust the threshold as needed
                result = "Your skin has wrinkles. :( "
            else:
                result = "Your skin has no wrinkles. :)"

            return render_template('index.html', result=result)
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
