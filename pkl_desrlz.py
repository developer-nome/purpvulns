import pickle
import base64
import os
from flask import Flask, request

app = Flask(__name__)


@app.route("/ingest", methods=["POST"])
def hackme():
    data = base64.urlsafe_b64decode(request.form['pickled'])
    deserialized = pickle.loads(data)
    os.system(deserialized)

    return '', 204