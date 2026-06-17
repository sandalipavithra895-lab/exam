import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1. TFLite Model එක සහ Labels Load කරගැනීම
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

# Model එක ලෝඩ් කරලා Allocating Tensors කරනවා
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Input සහ Output විස්තර ලබාගැනීම
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Colab එකේ දුන්න Image Size එක (180x180)
IMG_HEIGHT = 180
IMG_WIDTH = 180

# Labels ටික List එකකට කියවා ගැනීම
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
else:
    labels = []
    print("Warning: labels.txt file එක සොයාගත නොහැකි විය!")

# 2. API Endpoint එක නිර්මාණය කිරීම (/predict)
@app.route('/predict', methods=['POST'])
def predict():
    # ඇප් එකෙන් Image එකක් එවලා තියෙනවාද කියා බැලීම
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    
    try:
        # Image එක Open කරලා RGB බවට පත් කිරීම
        img = Image.open(file.stream).convert('RGB')
        
        # Model එකට ගැලපෙන Size එකට (180x180) Resize කිරීම
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Image එක Array එකක් බවට හරවා Normalize කිරීම (0-1 අතරට)
        input_data = np.array(img, dtype=np.float32) / 255.0
        
        # Batch dimension එකක් එකතු කිරීම (Shape එක [1, 180, 180, 3] කිරීමට)
        input_data = np.expand_dims(input_data, axis=0)
        
        # TFLite Model එකට Data ඇතුලත් කර Run කිරීම
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Result එක ලබාගැනීම
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # වැඩිම සම්භාවිතාවයක් (Highest Probability) තියෙන Category එක සෙවීම
        prediction_idx = int(np.argmax(output_data))
        confidence = float(output_data[prediction_idx])
        
        # Label එකක් තියෙනවා නම් නමද, නැත්නම් Index එකද ලබාදීම
        label = labels[prediction_idx] if prediction_idx < len(labels) else str(prediction_idx)
        
        # Android ඇප් එකට JSON එකක් විදිහට Result එක යැවීම
        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 3. Server එක Run කිරීම
if __name__ == '__main__':
    # Local Network එකේ ඇප් එකෙන් Connect වෙන්න පුළුවන් වෙන්න host='0.0.0.0' දෙනවා
    app.run(host='0.0.0.0', port=5000, debug=True)
