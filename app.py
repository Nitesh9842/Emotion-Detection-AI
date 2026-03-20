import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
from src.pipeline import EmotionPipeline

app = Flask(__name__)

# Caching the pipeline for fast subsequent requests
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            pipeline = EmotionPipeline.load(MODELS_DIR)
        except Exception:
            # Fallback if model doesn't exist yet, handle gracefully
            pass
    return pipeline

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    p = get_pipeline()
    if p is None:
        return jsonify({"error": "Models not found! Please run 'python main.py' to train models first."}), 500

    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    time_of_day = data.get("time_of_day", "morning")
    try:
        stress_level = float(data.get("stress_level", 3))
        energy_level = float(data.get("energy_level", 3))
        sleep_hours = float(data.get("sleep_hours", 7.0))
    except ValueError:
        return jsonify({"error": "Invalid numeric context features"}), 400
    
    try:
        result = p.predict(
            text,
            time_of_day=time_of_day,
            stress_level=stress_level,
            energy_level=energy_level,
            sleep_hours=sleep_hours
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch", methods=["POST"])
def batch():
    p = get_pipeline()
    if p is None:
        return jsonify({"error": "Models not found! Please run 'python main.py' to train models first."}), 500

    if "file" not in request.files and request.form.get("use_test_set") != "true":
        return jsonify({"error": "No file uploaded or test set selected"}), 400
        
    use_test_set = request.form.get("use_test_set") == "true"
    
    try:
        if use_test_set:
            test_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "arvyax_test_inputs_120.xlsx"
            )
            df = pd.read_excel(test_path)
        else:
            file = request.files["file"]
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
                
        if "journal_text" not in df.columns:
            return jsonify({"error": "The uploaded file must contain a 'journal_text' column."}), 400
            
        results_df = p.predict_batch(df)
        
        # Return CSV as downloadable file
        output = BytesIO()
        output.write(results_df.to_csv(index=False).encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output, 
            mimetype="text/csv", 
            as_attachment=True, 
            download_name="emotion_predictions.csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use reloader to auto-restart on code changes
    app.run(host="127.0.0.1", port=5000, debug=True)
