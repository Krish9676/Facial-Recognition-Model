# Installation for Windows - Dlib Issue Fix

## Problem
Dlib requires CMake to build from source, which is complex on Windows.

## Solution: Install Pre-built Wheels

Run these commands **in order**:

```bash
# 1. First install the packages that don't need dlib
pip install Flask Flask-CORS pymongo numpy Pillow opencv-python python-dotenv gunicorn

# 2. Install cmake first
pip install cmake

# 3. Try to install dlib-binary (pre-built wheel)
pip install dlib-binary

# 4. If dlib-binary fails, install dlib from conda-forge (if you have conda)
# Otherwise, download pre-built wheel manually

# 5. After dlib is installed, install face-recognition
pip install face-recognition
```

## Alternative: Download Pre-built Dlib Wheel

### Step 1: Check your Python version
```bash
python --version
```

### Step 2: Download pre-built wheel from:
- Go to: https://github.com/z-mahmud22/Dlib_Windows_Python3.x_package
- Download the wheel file matching your Python version
- For example, if Python 3.11: `dlib-19.22.99-cp311-cp311-win_amd64.whl`

### Step 3: Install the downloaded wheel
```bash
pip install path/to/downloaded/dlib-*.whl
```

### Step 4: Install remaining packages
```bash
pip install face-recognition
```

## Full Installation Script

```bash
# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install basic packages
pip install Flask==3.1.2 Flask-CORS==6.0.1 pymongo==4.15.3

# Install image processing
pip install numpy==2.2.6 Pillow==12.0.0 opencv-python==4.12.0.88

# Install utilities
pip install python-dotenv==1.2.1 gunicorn==23.0.0

# Install cmake for building
pip install cmake

# Try to install dlib (will still fail without proper build tools)
# But we'll skip this and use alternative method
# pip install dlib

# Instead, download dlib wheel from GitHub manually
# OR use pip install dlib-binary if available

# Finally install face-recognition
pip install face-recognition
```

## Quick Workaround (Test without dlib)

If you want to test the API without face recognition first:

1. Comment out face_recognition imports in functions.py temporarily
2. Test database connection and basic API endpoints
3. Then add face recognition later

## Best Solution: Use Conda

If you have Anaconda or Miniconda installed:

```bash
conda install -c conda-forge dlib
pip install face-recognition
pip install -r requirements.txt
```


