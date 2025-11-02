# Setup and Run Commands for Facial Recognition API

## Step 1: Check Python Version
```bash
python --version
```
Should be Python 3.8 or higher

## Step 2: Create Virtual Environment
```bash
python -m venv venv
```

## Step 3: Activate Virtual Environment

### On Windows (Git Bash/MINGW64):
```bash
source venv/Scripts/activate
```

### On Windows (Command Prompt):
```bash
venv\Scripts\activate
```

### On Linux/Mac:
```bash
source venv/bin/activate
```

## Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 5: Install Missing Dependencies (if needed)
```bash
pip install Flask==3.0.0
pip install Flask-CORS==4.0.0
pip install pymongo==4.6.0
pip install face-recognition==1.3.0
pip install numpy==1.24.3
pip install Pillow==10.1.0
pip install opencv-python==4.8.1.78
pip install python-dotenv==1.0.0
pip install gunicorn==21.2.0
```

## Step 6: Run the Application
```bash
python final.py
```

## Step 7: Test the API

### Open a new terminal (while keeping the server running)

### Test Health Endpoint:
```bash
curl http://localhost:5000/api/health
```

### Test Verification Endpoint (Python):
```bash
python
```
Then in Python:
```python
import requests
import base64

# Read and encode an image
with open('test_photo.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = requests.post('http://localhost:5000/api/verify', json={
    'photo': image_b64,
    'farm_name': 'PP-ELAI-29',
    'rsbsa_no': '14-32-13-040-000294'
})

print(response.json())
```

## Troubleshooting Commands

### Check if MongoDB is accessible:
```bash
python -c "from pymongo import MongoClient; MongoClient('mongodb+srv://elai_read_development:Elai%40developerRead2025@cluster0.et4eg.gcp.mongodb.net/').admin.command('ping')"
```

### Check installed packages:
```bash
pip list
```

### Deactivate virtual environment (when done):
```bash
deactivate
```

## Quick Start (All-in-One)
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python final.py
```


