import os
import base64
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import torch

app = Flask(__name__)
CORS(app)

# =========================================================
# 🔧 Configuration
# =========================================================
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "test")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "phil_farmers")
XOR_KEY = os.getenv("XOR_KEY", "MySecretKey123")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.6"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# =========================================================
# 🗄️ Database Initialization
# =========================================================
mongo_client = None
db = None
collection = None


def initialize_database():
    """Initialize MongoDB connection."""
    global mongo_client, db, collection
    try:
        print(f"🔌 Connecting to MongoDB...")
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        mongo_client.admin.command("ping")
        db = mongo_client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        print(f"✅ Connected to MongoDB: {MONGODB_DB_NAME}.{MONGODB_COLLECTION}")
        return True
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        mongo_client = db = collection = None
        return False


# =========================================================
# 🧩 Utility Functions
# =========================================================
def xor_decrypt(encrypted_data, key):
    decrypted = bytearray()
    key_bytes = key.encode("utf-8")
    for i, byte in enumerate(encrypted_data):
        decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
    return bytes(decrypted)


def decrypt_farmer_photo(encrypted_b64_string):
    try:
        encrypted_bytes = base64.b64decode(encrypted_b64_string)
        decrypted_bytes = xor_decrypt(encrypted_bytes, XOR_KEY)
        image = Image.open(BytesIO(decrypted_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"❌ Error decrypting photo: {e}")
        return None


def decode_uploaded_photo(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"❌ Error decoding uploaded photo: {e}")
        return None


def extract_face_embedding(image_pil):
    """Extract face embedding (lazy-load FaceNet models)."""
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False,
        )
        facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        face_tensor = mtcnn(image_pil)
        if face_tensor is None:
            print("⚠️ No face detected.")
            return None

        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0).to(device)
            embedding = facenet(face_tensor).cpu().numpy()[0]

        del mtcnn, facenet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embedding
    except Exception as e:
        print(f"❌ Error extracting face embedding: {e}")
        return None


def is_face_match(embedding1, embedding2, threshold=MATCH_THRESHOLD):
    try:
        distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
        is_match = distance <= threshold
        confidence = (1 - (distance / threshold)) * 100 if is_match else 0
        return is_match, distance, round(confidence, 2)
    except Exception as e:
        print(f"❌ Error comparing embeddings: {e}")
        return False, None, 0


# =========================================================
# 🌐 Routes
# =========================================================
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "running",
        "service": "Farmer Facial Recognition API",
        "version": "1.0",
        "device": str(device),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/health", methods=["GET"])
def health():
    db_status = "connected" if collection else "disconnected"
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "device": str(device)
    })


@app.route("/verify", methods=["POST"])
def verify_farmer():
    try:
        if collection is None:
            return jsonify({
                "verified": False,
                "message": "Database not initialized",
                "error_code": "DB_NOT_INITIALIZED"
            }), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        uploaded_photo_b64 = data.get("photo")
        farm_name = data.get("farm_name")
        rsbsa_no = data.get("rsbsa_no")

        if not all([uploaded_photo_b64, farm_name, rsbsa_no]):
            return jsonify({
                "verified": False,
                "message": "Missing required fields: photo, farm_name, rsbsa_no",
                "error_code": "MISSING_FIELDS"
            }), 400

        # 🔍 Find farmer in DB
        farmer = collection.find_one({
            "farm_name": farm_name.strip(),
            "RSBSA_no": rsbsa_no.strip()
        })

        if not farmer:
            return jsonify({
                "verified": False,
                "message": "Farmer not found in database",
                "error_code": "FARMER_NOT_FOUND"
            }), 404

        encrypted_info = farmer.get("encryptedData", {})
        if not encrypted_info or "encryptedContent" not in encrypted_info:
            return jsonify({
                "verified": False,
                "message": "No encrypted photo in database",
                "error_code": "NO_PHOTO_IN_DB"
            }), 404

        # 🖼️ Uploaded image
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

        # 🖼️ Stored image
        stored_image = decrypt_farmer_photo(encrypted_info["encryptedContent"])
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

        # ✅ Compare
        is_match, distance, confidence = is_face_match(stored_embedding, uploaded_embedding)

        return jsonify({
            "verified": is_match,
            "confidence": confidence,
            "distance": round(distance, 4) if distance else None,
            "threshold": MATCH_THRESHOLD,
            "farm_name": farm_name,
            "rsbsa_no": rsbsa_no,
            "farmer_name": farmer.get("name", "N/A"),
            "message": "Identity verified successfully" if is_match else "Identity verification failed",
            "verified_at": datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        print(f"❌ Error in /verify: {e}")
        return jsonify({
            "verified": False,
            "message": f"System error: {str(e)}",
            "error_code": "SYSTEM_ERROR"
        }), 500


# =========================================================
# 🚀 App Entry Point
# =========================================================
if __name__ == "__main__":
    if initialize_database():
        port = int(os.getenv("PORT", 8080))
        print(f"🚀 Server running on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        print("❌ Could not initialize MongoDB. Exiting...")
        exit(1)
