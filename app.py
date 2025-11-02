import os
import base64
import logging
import numpy as np
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import torch

# ---------- Flask / logging ----------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Config (from env) ----------
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'test')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'phil_farmers')
XOR_KEY = os.getenv('XOR_KEY', 'MySecretKey123')
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', '0.6'))

# ---------- Globals (to be initialized) ----------
mongo_client: Optional[MongoClient] = None
db = None
collection = None

# Models will be lazy-loaded and cached here
_mtcnn = None
_facenet = None

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# ---------- Database init ----------
def initialize_database() -> Tuple[Optional[MongoClient], Optional[object], Optional[object]]:
    """
    Initialize MongoDB connection and return (client, db, collection).
    If connection fails, returns (None, None, None).
    """
    global mongo_client, db, collection
    try:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI is not set")

        logging.info("Connecting to MongoDB...")
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        mongo_client.admin.command('ping')  # test connection
        db = mongo_client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]

        # optional: print count for debug (may be slow on very large collections)
        try:
            count = collection.count_documents({})
            logging.info(f"Connected to {MONGODB_DB_NAME}.{MONGODB_COLLECTION} (documents: {count})")
        except Exception:
            logging.info(f"Connected to {MONGODB_DB_NAME}.{MONGODB_COLLECTION} (count skipped)")

        return mongo_client, db, collection
    except Exception as e:
        logging.error(f"MongoDB connection failed: {e}")
        return None, None, None

# ---------- Image utilities ----------
def xor_decrypt(encrypted_data: bytes, key: str) -> bytes:
    decrypted = bytearray()
    key_bytes = key.encode('utf-8')
    for i, byte in enumerate(encrypted_data):
        decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
    return bytes(decrypted)

def decrypt_farmer_photo(encrypted_b64_string: str) -> Optional[Image.Image]:
    """
    Decrypt XOR-encrypted base64 image (as stored in DB).
    """
    try:
        encrypted_bytes = base64.b64decode(encrypted_b64_string)
        decrypted_bytes = xor_decrypt(encrypted_bytes, XOR_KEY)
        image = Image.open(BytesIO(decrypted_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logging.error(f"Error decrypting stored photo: {e}")
        return None

def decode_uploaded_photo(base64_string: str) -> Optional[Image.Image]:
    """Decode plain base64 uploaded image."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logging.error(f"Error decoding uploaded photo: {e}")
        return None

# ---------- Model loading ----------
def _load_models():
    """Lazy-load and cache MTCNN and Facenet models (first call)."""
    global _mtcnn, _facenet, device
    if _mtcnn is not None and _facenet is not None:
        return

    logging.info("Loading face detection and recognition models (this may take a while)...")
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        _mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False
        )
        _facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        logging.info("Models loaded and cached.")
    except Exception as e:
        logging.exception("Failed to load facenet models: %s", e)
        _mtcnn = None
        _facenet = None

def extract_face_embedding(image_pil: Image.Image) -> Optional[np.ndarray]:
    """Return 512-d embedding or None if no face / error."""
    try:
        # ensure models available
        if _mtcnn is None or _facenet is None:
            _load_models()
            if _mtcnn is None or _facenet is None:
                logging.error("Models not available.")
                return None

        face_tensor = _mtcnn(image_pil)
        if face_tensor is None:
            return None

        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0).to(device)
            embedding = _facenet(face_tensor).cpu().numpy()[0]
        return embedding
    except Exception as e:
        logging.exception("Error extracting face embedding: %s", e)
        return None

def is_face_match(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = MATCH_THRESHOLD):
    try:
        distance = float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
        is_match = distance <= threshold
        confidence = (1 - (distance / threshold)) * 100 if distance <= threshold else 0.0
        return bool(is_match), distance, round(float(confidence), 2)
    except Exception as e:
        logging.exception("Error comparing embeddings: %s", e)
        return False, None, 0.0

# ---------- Helper: fetch farmer ----------
def get_farmer(farm_name: str, rsbsa_no: str):
    """Return farmer document or None."""
    if collection is None:
        raise RuntimeError("Database collection not initialized")
    try:
        farmer = collection.find_one({
            "farm_name": farm_name.strip(),
            "RSBSA_no": rsbsa_no.strip()
        })
        return farmer
    except Exception as e:
        logging.exception("DB query failed: %s", e)
        return None

# ---------- Routes ----------
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "service": "Farmer Facial Recognition API",
        "version": "1.0",
        "device": str(device),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/health', methods=['GET'])
def health():
    db_status = "connected" if collection is not None else "disconnected"
    models_status = "loaded" if (_mtcnn is not None and _facenet is not None) else "not_loaded"
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "models": models_status,
        "device": str(device)
    })

@app.route('/verify', methods=['POST'])
def verify_farmer():
    """
    Expected JSON:
    { "photo": "<base64>", "farm_name": "...", "rsbsa_no": "..." }
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logging.error("Failed to parse JSON payload: %s", e)
        return jsonify({"error": "Invalid JSON payload"}), 400

    uploaded_photo_b64 = data.get('photo')
    farm_name = data.get('farm_name')
    rsbsa_no = data.get('rsbsa_no')

    if not all([uploaded_photo_b64, farm_name, rsbsa_no]):
        return jsonify({"error": "Missing required fields: photo, farm_name, rsbsa_no"}), 400

    # Ensure DB collection present
    if collection is None:
        logging.error("Attempt to verify while DB not initialized.")
        return jsonify({
            "verified": False,
            "message": "Database not initialized",
            "error_code": "DB_NOT_INITIALIZED"
        }), 500

    # Look up farmer
    try:
        farmer = get_farmer(farm_name, rsbsa_no)
    except RuntimeError as e:
        logging.error("DB not initialized: %s", e)
        return jsonify({
            "verified": False,
            "message": "Database not initialized",
            "error_code": "DB_NOT_INITIALIZED"
        }), 500

    if not farmer:
        return jsonify({
            "verified": False,
            "message": "Farmer not found in database",
            "error_code": "FARMER_NOT_FOUND"
        }), 404

    encdata = farmer.get('encryptedData', {})
    encrypted_content = encdata.get('encryptedContent')
    if not encrypted_content:
        return jsonify({
            "verified": False,
            "message": "No photo data for this farmer",
            "error_code": "NO_PHOTO_IN_DB"
        }), 404

    # Decode uploaded image
    uploaded_image = decode_uploaded_photo(uploaded_photo_b64)
    if uploaded_image is None:
        return jsonify({
            "verified": False,
            "message": "Invalid uploaded photo",
            "error_code": "INVALID_UPLOAD"
        }), 400

    uploaded_embedding = extract_face_embedding(uploaded_image)
    if uploaded_embedding is None:
        return jsonify({
            "verified": False,
            "message": "No face detected in uploaded photo",
            "error_code": "NO_FACE_IN_UPLOAD"
        }), 400

    # Decrypt and get stored photo
    stored_image = decrypt_farmer_photo(encrypted_content)
    if stored_image is None:
        return jsonify({
            "verified": False,
            "message": "Failed to decrypt stored photo",
            "error_code": "DECRYPTION_FAILED"
        }), 500

    stored_embedding = extract_face_embedding(stored_image)
    if stored_embedding is None:
        return jsonify({
            "verified": False,
            "message": "No face detected in stored photo",
            "error_code": "NO_FACE_IN_DB"
        }), 500

    # Compare
    is_match, distance, confidence = is_face_match(stored_embedding, uploaded_embedding)

    response = {
        "verified": is_match,
        "confidence": confidence,
        "distance": round(distance, 4) if distance is not None else None,
        "threshold": MATCH_THRESHOLD,
        "farm_name": farm_name,
        "rsbsa_no": rsbsa_no,
        "farmer_name": farmer.get('name', 'N/A'),
        "message": "Identity verified successfully" if is_match else "Identity verification failed",
        "verified_at": datetime.utcnow().isoformat() + "Z"
    }
    return jsonify(response), 200

# ---------- Startup ----------
if __name__ == "__main__":
    client, db, collection = initialize_database()
    if collection is None:
        logging.error("Database initialization failed on startup. Exiting.")
        # Exit so Render shows failure logs instead of running without DB.
        raise SystemExit("DB init failed")
    # Start Flask (development server). Render uses Gunicorn via the Dockerfile/CMD.
    port = int(os.getenv('PORT', 8080))
    logging.info(f"Starting Flask dev server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
