import os
import time
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from preprocessor import preprocess_canvas_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Cargar modelos entrenados
models = {}
model_paths = {
    "svm": os.path.join(MODELS_DIR, "best_svm_model.joblib"),
    "mlp": os.path.join(MODELS_DIR, "best_mlp_model.joblib"),
    "rf": os.path.join(MODELS_DIR, "best_rf_model.joblib")
}

for key, path in model_paths.items():
    if os.path.exists(path):
        models[key] = joblib.load(path)
        print(f"Modelo cargado: {key.upper()} desde {path}")
    else:
        print(f"Advertencia: No se encontró el modelo {key} en {path}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/info", methods=["GET"])
def info():
    return jsonify({
        "project": "Digits - Reconocimiento Óptico de Dígitos Manuscritos",
        "author": "José Alejandro Bernal Figueroa",
        "career": "Ingeniería de Sistemas",
        "models_available": list(models.keys()),
        "description": "Sistema de reconocimiento de dígitos manuscritos con Machine Learning (UCI Digits 8x8) y lienzo interactivo de dibujo en tiempo real."
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_data = data.get("image")
        model_name = data.get("model", "svm").lower()

        if not image_data:
            return jsonify({"error": "No se proporcionó imagen"}), 400

        selected_model = models.get(model_name)
        if not selected_model:
            selected_model = models.get("svm")
            model_name = "svm"

        t_start = time.time()

        # Preprocesar imagen del canvas a vector 1x64 y matriz 8x8
        features_flat, matrix_8x8 = preprocess_canvas_image(image_data)

        # Si el lienzo está vacío (todo ceros)
        if matrix_8x8.max() == 0:
            return jsonify({
                "digit": None,
                "confidence": 0.0,
                "probabilities": [0.0] * 10,
                "matrix_8x8": matrix_8x8.tolist(),
                "message": "Lienzo vacío. Por favor dibuja un dígito del 0 al 9."
            })

        # Predicción e inferencia de probabilidades
        prediction = int(selected_model.predict(features_flat)[0])

        if hasattr(selected_model, "predict_proba"):
            probs = selected_model.predict_proba(features_flat)[0]
            confidence = float(probs[prediction])
            prob_list = [float(p) for p in probs]
        else:
            prob_list = [0.0] * 10
            prob_list[prediction] = 1.0
            confidence = 1.0

        latency_ms = round((time.time() - t_start) * 1000, 2)

        return jsonify({
            "success": True,
            "digit": prediction,
            "confidence": round(confidence * 100, 2),
            "probabilities": [round(p * 100, 2) for p in prob_list],
            "matrix_8x8": matrix_8x8.tolist(),
            "model_used": model_name.upper(),
            "latency_ms": latency_ms
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Servidor Digits iniciado en http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
