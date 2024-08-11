# Install flask if you don't have it
!pip install flask==2.2.3

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and text vectorization
model = tf.keras.models.load_model('path/to/your/saved/model')  # Replace with actual path
text_vectorization = joblib.load("text_vectorization.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    if text:
        text_vectorized = text_vectorization([text])
        prediction = model.predict(text_vectorized)
        predicted_class = np.argmax(prediction[0])
        return jsonify({'sentiment': int(predicted_class)})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
