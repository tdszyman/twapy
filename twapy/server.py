"""Twapy demo server. This is a basic Flask app to serve an index page and to solve analogies via
GET requests to the /analogy/ endpoint.

To run the server locally, execute the runserver.bat or runserver.sh script.

"""

from flask import Flask, jsonify, render_template

from .models import ModelCollection
from .alignment import Analogy

# Set the directory containing the embedding models here:
model_directory = "models"
collection = ModelCollection(model_directory)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analogy/<model1>/<model2>/<word>')
def analogy(model1, model2, word):
    # Calculate the analogy result
    a = Analogy(word, model1, model2, collection=collection)
    obj = {
        "message": "Word {:} mapped from {:} to {:}".format(word, model1, model2),
        "word2": a.word2
    }
    return jsonify(obj)
