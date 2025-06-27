from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
from fraud_detection import FraudDetectionSystem
import os
import sys
from datetime import datetime
from flask import request

from dotenv import load_dotenv
load_dotenv()




key = os.getenv("KEY")
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize fraud detection system
fraud_system = FraudDetectionSystem()
try:
    fraud_system.load_model('fraud_detection_model.joblib')
    print("Model loaded from fraud_detection_model.joblib")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please ensure fraud_detection_model.joblib exists in the current directory")

def format_time(seconds):
    """Convert seconds to readable time format"""
    return datetime.fromtimestamp(seconds).strftime('%H:%M:%S')
import requests

import requests

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    headers = {
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "inputs": f"User: {user_message}\nAssistant:",
        "parameters": {"max_new_tokens": 128}
    }
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers=headers,
            json=payload
        )
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            reply = data[0]["generated_text"].split("Assistant:")[-1].strip()
        elif "error" in data:
            reply = f"Error from Hugging Face: {data['error']}"
        else:
            reply = str(data)
    except Exception as e:
        reply = f"Error communicating with Hugging Face: {str(e)}"
    return jsonify({'reply': reply})
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Read the CSV file
            try:
                df = pd.read_csv(filepath)
                
                # Perform fraud detection
                try:
                    # Select numeric columns and handle missing values
                    numeric_data = df.select_dtypes(include=[np.number])
                    if 'Class' in numeric_data.columns:
                        numeric_data = numeric_data.drop('Class', axis=1)
                    
                    # Fill missing values with mean
                    numeric_data = numeric_data.fillna(numeric_data.mean())
                    
                    # Make predictions
                    predictions, probabilities = fraud_system.predict(numeric_data)
                    
                    # Prepare data for display
                    results_data = []
                    for idx, row in df.iterrows():
                        result = {
                            'Time': format_time(row['Time']),
                            'Amount': row['Amount'],
                            'prediction': int(predictions[idx]),
                            'probability': float(probabilities[idx]),
                            'features': [col for col in df.columns if col not in ['Time', 'Amount', 'Class']]
                        }
                        # Add feature values
                        for feature in result['features']:
                            result[feature] = row[feature]
                        results_data.append(result)
                    
                    # Calculate statistics
                    total_transactions = len(df)
                    fraud_count = sum(predictions)
                    legitimate_count = total_transactions - fraud_count
                    total_amount = df['Amount'].sum()
                    
                    # Calculate percentages
                    fraud_percentage = (fraud_count / total_transactions * 100)
                    legitimate_percentage = (legitimate_count / total_transactions * 100)
                    
                    # Pagination
                    page = request.args.get('page', 1, type=int)
                    per_page = 10
                    total_pages = (total_transactions + per_page - 1) // per_page
                    start_idx = (page - 1) * per_page
                    end_idx = start_idx + per_page
                    
                    return render_template('display.html',
                                        data=results_data[start_idx:end_idx],
                                        total_transactions=total_transactions,
                                        fraud_count=fraud_count,
                                        legitimate_count=legitimate_count,
                                        fraud_percentage=round(fraud_percentage, 1),
                                        legitimate_percentage=round(legitimate_percentage, 1),
                                        total_amount="{:,.2f}".format(total_amount),
                                        current_page=page,
                                        total_pages=total_pages)
                
                except Exception as e:
                    flash(f"Error in fraud detection: {str(e)}")
                    print(f"Error in fraud detection: {str(e)}")
                    return redirect(url_for('index'))
                    
            except Exception as e:
                flash(f"Error reading CSV file: {str(e)}")
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f"Error saving file: {str(e)}")
            return redirect(url_for('index'))
    
    flash('Please upload a valid CSV file')
    return redirect(url_for('index'))

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    flash('An internal error occurred. Please try again.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        print("Starting Flask application...")
        print(f"Debug mode: {'on' if app.debug else 'off'}")
        print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
        print(f"Templates folder: {app.template_folder}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting Flask application: {str(e)}")
        sys.exit(1) 