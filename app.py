from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os
import time
from model import train_selected_model, train_ensemble_model

app = Flask(__name__)
CORS(app)

TARGET_NAMES = ['Low', 'Medium', 'High']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.json
    algorithms = data.get("algorithms")

    if not algorithms or not isinstance(algorithms, list) or len(algorithms) == 0:
        return jsonify({"error": "Please select at least one algorithm."}), 400

    try:
        if len(algorithms) == 1:
            model, accuracy, training_time, class_report, plot_path = train_selected_model(algorithms[0])
            model_name = algorithms[0].replace('_', ' ').title()
            message = f"{model_name} trained successfully!"
        else:
            model, accuracy, training_time, class_report, plot_path = train_ensemble_model(algorithms)
            model_names = [alg.replace('_', ' ').title() for alg in algorithms]
            message = f"Ensemble model ({', '.join(model_names)}) trained successfully!"

        joblib.dump(model, "model.pkl")
        
        timestamp = int(time.time())
        confusion_matrix_url = f'/{plot_path}?v={timestamp}'

        return jsonify({
            "message": message,
            "accuracy": accuracy,
            "training_time": training_time,
            "classification_report": class_report,
            "confusion_matrix_url": confusion_matrix_url
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists("model.pkl"):
        return jsonify({"error": "Model not trained yet. Please train a model first."}), 400

    try:
        data = request.json
        advertising_spend = float(data["advertising_spend"])
        promotions = int(data["promotions"])
        holidays = int(data["holidays"])

        model = joblib.load("model.pkl")
        
        prediction = model.predict([[advertising_spend, promotions, holidays]])[0]
        
        predicted_label = TARGET_NAMES[prediction]

        return jsonify({
            "predicted_sales_volume": predicted_label
        })
    except (KeyError, TypeError):
        return jsonify({"error": "Invalid input data. Please provide 'advertising_spend', 'promotions', and 'holidays'."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
