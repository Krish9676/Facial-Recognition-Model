# Installation Steps for Windows

## Step 1: Upgrade pip and setuptools
```bash
python -m pip install --upgrade pip setuptools wheel
```

## Step 2: Install system dependencies (if needed)

### Option A: Install dlib from pre-built wheel (Recommended for Windows)
```bash
pip install dlib-binary
```

### Option B: Build from source (slower)
```bash
pip install cmake
pip install dlib
```

## Step 3: Install main dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Install face-recognition
```bash
pip install face-recognition
```

## Step 5: Verify installation
```bash
python -c "import face_recognition; print('Face recognition installed successfully!')"
```

## Alternative: Install packages one by one
```bash
pip install Flask>=3.0.0
pip install Flask-CORS>=4.0.0
pip install pymongo>=4.6.0
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install opencv-python>=4.8.0
pip install python-dotenv>=1.0.0
pip install gunicorn>=21.2.0
pip install dlib-binary
pip install face-recognition>=1.3.0
```

## Troubleshooting

### If dlib installation fails:
1. Install Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Or download pre-built wheel from: https://github.com/z-mahmud22/Dlib_Windows_Python3.x_package

### If face-recognition fails:
```bash
pip install --upgrade face-recognition
```

### Check what's installed:
```bash
pip list
```



