from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/predict", methods=["GET", "POST"])
def predict_api():
    req = request.get_json(force=True)

    if type(req) == dict:
        X = np.array(list(req.values())).reshape(-1, 13)
        prediction = model.predict(X)[0]
        return jsonify({"status": prediction.tolist()})

    elif type(req) == list:
        X = []
        for el in req:
            values = list(el.values())
            data = np.array(values).reshape(1, -1)

            X.append(data)
        X = np.array(X, dtype=np.double).reshape(-1, 13)

        prediction = model.predict(X)

        return jsonify({"statuses": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
