
import os
import numpy as np
import tkinter as tk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from datetime import datetime
import logging
from google.oauth2 import service_account
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QNQpredictor")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Load the trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'Model', 'QNQupdated_model_overall17.keras')
model = load_model(model_path)

# Google Sheets parameters
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1WnQ3_QtX9r7VmR6jYVu4bFgW8FV4OX8a7821KpEU4PE"
RANGE_NAME = "Sheet1!A:H"  # Adjust the range according to your sheet

# Function to authenticate and connect to Google Sheets
def connect_to_google_sheets():
    # Update the path to the new credentials file
    credentials_path = os.path.join(base_dir, 'BeemoApp', 'GsheetAPI', 'QNQ-New-Client.json')
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    creds.refresh(Request())
    try:
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        return sheet
    except Exception as err:
        logger.error(err)
        return None

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    if img_path is None:
        logger.error("Image path is None")
        raise ValueError("Image path is None")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to collect new data and labels for retraining
def collect_new_data_and_labels(true_label, img_path):
    logger.info(f"Collecting new data for retraining. img_path: {img_path}, true_label: {true_label}")
    img_array = load_and_preprocess_image(img_path)
    new_data = img_array
    new_labels = np.array([true_label])
    return new_data, new_labels

# Function to retrain the model incrementally
def retrain_qnq_model(model, new_data, new_labels):
    logger.info("Starting retraining of QNQ model...")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(new_data, new_labels, epochs=1, verbose=0)
    model.save(model_path)  # Save the updated model
    logger.info(f"QNQ model retrained and saved to {model_path}")

# Function to save results to Google Sheets
def QNQ_save_results_to_google_sheets(img_path, true_label, predicted_class, confidence, f1, precision, model_retrained=False):
    # Get current date and time
    current_time = datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    time_str = current_time.strftime("%H:%M:%S")

    # Convert all values to standard Python types
    confidence = float(confidence)  # Ensure confidence is a float
    f1 = float(f1)  # Ensure f1 is a float
    precision = float(precision)  # Ensure precision is a float
    
    sheet = connect_to_google_sheets()
    if sheet:
        values = [[date_str, time_str, os.path.basename(img_path), true_label, predicted_class, confidence * 100, f1, precision, "Yes" if model_retrained else "No"]]
        body = {'values': values}
        result = sheet.values().append(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
            valueInputOption="RAW", body=body).execute()
        logger.info(f"{result.get('updates').get('updatedCells')} cells appended to Google Sheet.")

# Function to predict and display results for a specific image
def QNQpredictor(img_path, output_box, threshold=0.85):
    # Define class names
    class_names = ['No Queen Detected', 'Queen Detected']

    # Initialize lists to store true labels and predictions
    true_labels = []
    predictions = []

    # Load and preprocess the specific image
    img_array = load_and_preprocess_image(img_path)

    # Predict the class of the image
    pred = model.predict(img_array)
    confidence = pred[0][0]
    predicted_class = 1 if confidence > threshold else 0

    # Display prediction in output box
    if isinstance(output_box, tk.Text):
        output_box.insert(tk.END, f'File: {os.path.basename(img_path)}\nPredicted: {class_names[predicted_class]}\nConfidence: {confidence * 100:.2f}%\n')

    # Assume a default true label for this example
    true_label = 1  # Modify as needed
    true_labels.append(true_label)
    predictions.append(predicted_class)

    # Calculate metrics
    f1 = f1_score(true_labels, predictions, zero_division=1)
    precision = precision_score(true_labels, predictions, zero_division=1)
    QNQ_save_results_to_google_sheets(img_path, true_labels[-1], predicted_class, confidence, f1, precision, model_retrained=False)

    return predicted_class, confidence, f1, precision

# Function to handle feedback and retrain if prediction is incorrect
def handle_feedback(img_path, output_box, feedback):
    predicted_class, confidence, f1, precision = QNQpredictor(img_path, output_box)
    if feedback == "incorrect":
        true_label = ask_true_label()  # Ask for manual input of the true label
        new_data, new_labels = collect_new_data_and_labels(true_label, img_path)
        retrain_qnq_model(model, new_data, new_labels)
        QNQ_save_results_to_google_sheets(img_path, true_label, predicted_class, confidence, f1, precision, model_retrained=True)
        output_box.insert(tk.END, "Retraining completed based on feedback.\n")
    else:
        QNQ_save_results_to_google_sheets(img_path, true_label, predicted_class, confidence, f1, precision, model_retrained=False)
        output_box.insert(tk.END, "No retraining needed based on feedback.\n")

# # Function to ask for manual input of the true label
def ask_true_label():
    import tkinter.simpledialog
    true_label = tkinter.simpledialog.askinteger("Input", "Enter the true label (0 for No Queen Detected, 1 for Queen Detected):")
    return true_label


