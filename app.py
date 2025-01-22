from flask import Flask, request, jsonify, send_from_directory
import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)


def load_intents():
    with open("moshaver.json", encoding="utf-8") as f:
        return json.load(f)

intents = load_intents()["intents"]


patterns = []
responses = []
tags = []

for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        responses.append(intent["responses"])
        tags.append(intent["tag"])


tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(X, padding="post")


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)


MODEL_FILE = "chatbot_model.h5"
if not os.path.exists(MODEL_FILE):
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(tags)), activation="softmax"))

    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    X = np.expand_dims(X, axis=-1)

   
    model.fit(X, y, epochs=500, batch_size=8, verbose=1)

    
    model.save(MODEL_FILE)
    print("Model trained for 500 epochs and saved.")
else:
  
    model = load_model(MODEL_FILE)
    print("Model loaded from file.")


def get_response(user_message):
  
    seq = tokenizer.texts_to_sequences([user_message])
    seq = pad_sequences(seq, padding="post", maxlen=X.shape[1])

   
    pred = model.predict(np.expand_dims(seq, axis=-1))
    tag = label_encoder.inverse_transform([np.argmax(pred)])

    
    for intent in intents:
        if intent["tag"] == tag[0]:
            return random.choice(intent["responses"])

    return "متأسفم، پاسخ مناسبی برای سؤال شما پیدا نکردم."


@app.route("/")
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    if not user_message:
        return jsonify({"response": "لطفاً یک پیام وارد کنید."})

    response = get_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
