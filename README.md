# Agro-Aqua-AI

Agro-Aqua-AI is a Flask-based AI web application for agriculture and aquaculture decision support. It combines multiple machine learning models, user authentication, a dashboard, prediction history, QR-based profile/report features, and separate modules for crop and aqua use cases.

## Features

* User login system with session-based access control
* Dashboard with prediction stats, recent activity, and reports
* Profile page with QR code generation
* Prediction history and report tracking
* Crop recommendation
* Crop yield prediction
* Crop water test / potable water check
* Plant disease detection
* Fish life prediction
* Water quality prediction

## Tech Stack

* **Backend:** Flask
* **ML/Data:** NumPy, Joblib, scikit-learn
* **QR features:** qrcode[pil], Pillow
* **Deployment:** Gunicorn
* **Frontend:** HTML, CSS, Flask templates
* **Storage:** Database-backed user/history/report system

## Project Structure

```bash
Agro-Aqua-AI/
├── app.py
├── requirements.txt
├── README.md
├── static/
├── templates/
├── farming models/
├── aqua models/
├── farming ml datasets/
├── aqua farm ml datasets/
├── best_farm_models/
└── best_aqua_models/
```

## How It Works

1. The user logs in.
2. The app loads the correct model and scaler with Joblib.
3. Form inputs are converted to floats.
4. Inputs are reshaped into a 2D NumPy array.
5. The scaler transforms the values.
6. The model predicts the result.
7. The result is shown on the page and saved to history/reports.

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

If you are deploying to production, use Gunicorn.

## Requirements

The repository uses:

* flask
* numpy
* joblib
* scikit-learn
* qrcode[pil]
* pillow
* gunicorn

## Notes

* The app uses saved `.joblib` model and scaler files.
* Inputs must match the exact feature order expected by each model.
* Scaling is required before prediction because the models were trained on scaled data.

## Future Improvements

* Add charts to the dashboard
* Add model performance pages
* Add API endpoints
* Add better error handling and input validation
* Add responsive UI improvements

## Author

Srikhar07
