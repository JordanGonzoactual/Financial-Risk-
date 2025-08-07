import logging
import sys
import os
import io
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError

# Add project root to Python path to allow imports from other directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the singleton model service
# This will initialize the service and load the model on startup
from backend.model_service import model_service
from shared.models.raw_data import RawData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

    # --- Request Logging ---
    @app.before_request
    def log_request_info():
        logging.info(f'Request: {request.method} {request.path} from {request.remote_addr}')
        if request.is_json and request.get_json(silent=True) is not None:
            logging.debug(f'Request JSON: {request.get_json()}')

    # --- API Endpoints ---
    @app.route('/health', methods=['GET'])
    def health_check():
        """Endpoint to check the service's health and model status."""
        health = model_service.health_check()
        if health['model_loaded']:
            return jsonify({"status": "ok", "message": "Service is healthy and model is loaded."}), 200
        else:
            return jsonify({"status": "error", "message": "Service is unhealthy, model not loaded."}), 503

    @app.route('/model-info', methods=['GET'])
    def get_model_info():
        """Endpoint to get metadata about the loaded model."""
        info = model_service.get_model_info()
        if "error" in info:
            return jsonify(info), 503
        return jsonify(info), 200

    @app.route('/', methods=['GET'])
    def index():
        return send_from_directory('../frontend', 'index.html')

    @app.route('/static/<path:path>')
    def send_static(path):
        return send_from_directory('../frontend/static', path)

    @app.route('/validate', methods=['POST'])
    def validate():
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        
        try:
            # Strict validation using RawData model
            RawData.model_validate(data)
            return jsonify({"status": "valid", "message": "Data validation passed"}), 200
        except ValidationError as e:
            return jsonify({
                "error": "ValidationError", 
                "details": str(e)
            }), 400

    @app.route('/predict', methods=['POST'])
    def predict():
        """Endpoint to receive batch loan data and return risk predictions."""
        logging.info("--- New Prediction Request Received ---")

        # 1. Content-Type Validation
        if 'text/csv' not in request.content_type:
            logging.warning(f"Content-Type check failed. Expected 'text/csv', got '{request.content_type}'.")
            return jsonify({"error": "Invalid Content-Type"}), 400
        logging.info("Content-Type check passed.")

        try:
            # 2. CSV Parsing
            csv_data = request.data.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))
            logging.info(f"CSV parsed successfully with {len(df)} records.")

            # 3. Strict Schema Validation
            # Validate each row using RawData model
            records = df.to_dict(orient='records')
            validated_records = []
            
            for i, record in enumerate(records):
                try:
                    validated_record = RawData.model_validate(record)
                    # Use by_alias=True to get alias names that the pipeline expects
                    validated_records.append(validated_record.model_dump(by_alias=True))
                except ValidationError as e:
                    logging.error(f"Validation error at row {i}: {str(e)}")
                    return jsonify({
                        "error": "ValidationError",
                        "message": f"Data validation failed at row {i}",
                        "details": str(e)
                    }), 400
            
            logging.info("Strict schema validation complete.")
            
            # 4. Inference Pipeline (using validated data)
            validated_df = pd.DataFrame(validated_records)
            
            # The model_service's predict_batch method will handle the pipeline transformation
            logging.info("Calling inference pipeline and making predictions with validated data...")
            predictions = model_service.predict_batch(validated_df)
            logging.info("Inference and prediction call successful.")

            # 5. Construct Response with Predictions
            response_data = {
                "predictions": predictions.tolist()
            }
            return jsonify(response_data), 200

        except (pd.errors.EmptyDataError, ValueError) as e:
            logging.error("Validation error during prediction", exc_info=True)
            error_message = str(e)
            details = {}
            if "Missing required columns" in error_message:
                try:
                    missing_cols_str = error_message.split(":", 1)[1].strip()
                    details["missing_columns"] = eval(missing_cols_str)
                except Exception:
                    details["raw_message"] = error_message
            else:
                details["raw_message"] = error_message

            return jsonify({
                "error": "ValidationError",
                "message": "The provided data does not match the required schema.",
                "details": details
            }), 400
        except TypeError as e:
            logging.error("Type error during prediction", exc_info=True)
            return jsonify({
                "error": "TypeError",
                "message": "A data type issue occurred during processing.",
                "details": {"raw_message": str(e)}
            }), 400
        except Exception as e:
            logging.critical("An unexpected error occurred in the prediction workflow", exc_info=True)
            return jsonify({
                "error": "InternalServerError",
                "message": "An unexpected error occurred on the server.",
                "details": {"raw_message": str(e)}
            }), 500

    return app

# --- Main Execution ---
if __name__ == '__main__':
    # The model_service is initialized when its module is imported.
    # We just need to check its status to decide if we should run the app.
    app = create_app()
    if model_service.health_check()['model_loaded']:

        logging.info("Starting Flask development server.")
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        logging.critical("Application failed to start because the model could not be loaded.")
