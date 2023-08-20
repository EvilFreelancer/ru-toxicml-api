from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from toxic_model import ToxicModel

app = Flask(__name__)

toxic = ToxicModel()

@app.on_event('startup')
def startup():
    logging.basicConfig(format='%(levelname)s:\t %(message)s', level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    inputs = toxic.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = toxic.model(**inputs).logits
    probs = logits.softmax(dim=1)

    # print(probs)
    neutral = probs[0][0].item()
    toxic = probs[0][1].item()

    return jsonify({
        "text": text,
        "prediction": {
            "neutral": neutral,
            "toxic": toxic
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
