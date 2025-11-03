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
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)
CORS(app)

# ====================================================================
# CONFIGURATION
# ====================================================================
MONGODB_URI = os.getenv('MONGODB_URI') or os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'test')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'phil_farmers')
XOR_KEY = os.getenv('XOR_KEY', 'MySecretKey123')
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', '0.6'))
MAX_IMAGE_SIZE = (800, 800)  # Resize large images

print(f"📋 Configuration:")
print(f"  MongoDB URI: {MONGODB_URI[:50] + '...' if MONGODB_URI else 'Not set'}")
print(f"  DB Name: {MONGODB_DB_NAME}")
print(f"  Collection: {MONGODB_COLLECTION}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ====================================================================
# GLOBAL MODEL INITIALIZATION (CRITICAL OPTIMIZATION)
# ====================================================================
print("🔄 Loading face recognition models (one-time initialization)...")
mtcnn = None
facenet = None

def initialize_models():
    """Initialize models ONCE at startup"""
    global mtcnn, facenet
    try:
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False
        )
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print("✅ Models loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

# ====================================================================
# DATABASE INITIALIZATION
# ====================================================================
mongo_client = None
db = None
collection = None

def initialize_database():
    """Initialize MongoDB connection"""
    global mongo_client, db, collection
    
    try:
        if not MONGODB_URI:
            raise ValueError("MongoDB URI not set")
        
        print(f"🔗 Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB server connection successful")
        
        # Set database and collection
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        mongo_client = client
        
        # Count documents
        try:
            count = collection.count_documents({})
            print(f"✅ Connected to database '{MONGODB_DB_NAME}'")
            print(f"✅ Collection '{MONGODB_COLLECTION}' has {count} farmers")
        except Exception as e:
            print(f"⚠️ Connected but couldn't count documents: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print(f"❌ Error details: {type(e).__name__}: {str(e)}")
        mongo_client = db = collection = None
        return False

# Initialize at module load for Gunicorn
print("🚀 Initializing database at module load...")
db_initialized = initialize_database()
if not db_initialized:
    print("❌ WARNING: Database initialization failed!")
else:
    print("✅ Database initialization complete")

# Initialize models at startup (CRITICAL)
models_initialized = initialize_models()
if not models_initialized:
    print("❌ WARNING: Model initialization failed!")
else:
    print("✅ Model initialization complete")

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================
def optimize_image(image_pil):
    """Resize image if too large to speed up processing"""
    if image_pil.size[0] > MAX_IMAGE_SIZE[0] or image_pil.size[1] > MAX_IMAGE_SIZE[1]:
        image_pil.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    return image_pil

def xor_decrypt(encrypted_data, key):
    """Simple XOR decryption"""
    decrypted = bytearray()
    key_bytes = key.encode('utf-8')
    for i, byte in enumerate(encrypted_data):
        decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
    return bytes(decrypted)

def decrypt_farmer_photo(encrypted_b64_string):
    """Decrypt XOR-encrypted base64 image from DB"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_b64_string)
        decrypted_bytes = xor_decrypt(encrypted_bytes, XOR_KEY)
        image = Image.open(BytesIO(decrypted_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = optimize_image(image)
        return image
    except Exception as e:
        print(f"❌ Image decryption failed: {e}")
        return None

def decode_uploaded_photo(base64_string):
    """Decode uploaded base64 photo"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = optimize_image(image)
        return image
    except Exception as e:
        print(f"❌ Photo decoding failed: {e}")
        return None

def extract_face_embedding(image_pil):
    """Extract face embedding using pre-loaded models"""
    global mtcnn, facenet
    
    if mtcnn is None or facenet is None:
        print("❌ Models not initialized")
        return None
    
    try:
        face_tensor = mtcnn(image_pil)
        if face_tensor is None:
            return None
        
        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0).to(device)
            embedding = facenet(face_tensor).cpu().numpy()[0]
        
        return embedding
        
    except Exception as e:
        print(f"❌ Face embedding extraction failed: {e}")
        return None

def is_face_match(embedding1, embedding2, threshold=MATCH_THRESHOLD):
    """Determine if faces match"""
    try:
        distance = float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
        is_match = bool(distance <= threshold)
        confidence = float((1 - (distance / threshold)) * 100) if is_match else 0.0
        return is_match, distance, round(confidence, 2)
    except Exception as e:
        print(f"❌ Face matching failed: {e}")
        return False, None, 0.0

# ====================================================================
# ROUTES
# ====================================================================
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "Farmer Facial Recognition API",
        "version": "1.0",
        "device": str(device),
        "database_connected": collection is not None,
        "models_loaded": mtcnn is not None and facenet is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    db_status = "connected" if collection is not None else "disconnected"
    model_status = "loaded" if (mtcnn is not None and facenet is not None) else "not_loaded"
    
    doc_count = None
    if collection is not None:
        try:
            doc_count = collection.count_documents({})
        except Exception as e:
            print(f"Error counting documents: {e}")
    
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "models": model_status,
        "document_count": doc_count,
        "device": str(device),
        "mongodb_uri_set": bool(MONGODB_URI and MONGODB_URI != 'mongodb://localhost:27017/')
    })

@app.route('/verify', methods=['POST'])
def verify_farmer():
    """Verify farmer identity using facial recognition"""
    
    if collection is None:
        print("❌ Database not initialized")
        return jsonify({
            "verified": False,
            "message": "Database not initialized",
            "error_code": "DB_NOT_INITIALIZED"
        }), 500
    
    if mtcnn is None or facenet is None:
        print("❌ Models not initialized")
        return jsonify({
            "verified": False,
            "message": "Models not initialized",
            "error_code": "MODELS_NOT_INITIALIZED"
        }), 500
    
    try:
        start_time = datetime.now()
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        uploaded_photo_b64 = data.get('photo')
        farm_name = data.get('farm_name')
        rsbsa_no = data.get('rsbsa_no')
        
        if not all([uploaded_photo_b64, farm_name, rsbsa_no]):
            return jsonify({
                "error": "Missing required fields: photo, farm_name, rsbsa_no"
            }), 400
        
        print(f"\n🔍 Verifying: {farm_name} | {rsbsa_no}")
        
        # Query database
        query_start = datetime.now()
        farmer = collection.find_one({
            "farm_name": farm_name.strip(),
            "RSBSA_no": rsbsa_no.strip()
        })
        query_time = (datetime.now() - query_start).total_seconds()
        print(f"⏱️ DB query: {query_time:.2f}s")
        
        if not farmer:
            print("❌ Farmer not found")
            return jsonify({
                "verified": False,
                "message": "Farmer not found in database",
                "error_code": "FARMER_NOT_FOUND"
            }), 404
        
        print(f"✅ Found: {farmer.get('name', 'N/A')}")
        
        if 'encryptedData' not in farmer or 'encryptedContent' not in farmer['encryptedData']:
            return jsonify({
                "verified": False,
                "message": "No photo data for this farmer",
                "error_code": "NO_PHOTO_IN_DB"
            }), 404
        
        # Process uploaded photo
        decode_start = datetime.now()
        uploaded_image = decode_uploaded_photo(uploaded_photo_b64)
        if uploaded_image is None:
            return jsonify({
                "verified": False,
                "message": "Invalid uploaded photo",
                "error_code": "INVALID_UPLOAD"
            }), 400
        decode_time = (datetime.now() - decode_start).total_seconds()
        print(f"⏱️ Decode upload: {decode_time:.2f}s")
        
        # Extract uploaded face embedding
        embed_start = datetime.now()
        uploaded_embedding = extract_face_embedding(uploaded_image)
        if uploaded_embedding is None:
            return jsonify({
                "verified": False,
                "message": "No face detected in uploaded photo",
                "error_code": "NO_FACE_IN_UPLOAD"
            }), 400
        embed_time = (datetime.now() - embed_start).total_seconds()
        print(f"⏱️ Extract upload embedding: {embed_time:.2f}s")
        
        # Process stored photo
        decrypt_start = datetime.now()
        stored_image = decrypt_farmer_photo(farmer['encryptedData']['encryptedContent'])
        if stored_image is None:
            return jsonify({
                "verified": False,
                "message": "Failed to decrypt stored photo",
                "error_code": "DECRYPTION_FAILED"
            }), 500
        decrypt_time = (datetime.now() - decrypt_start).total_seconds()
        print(f"⏱️ Decrypt DB photo: {decrypt_time:.2f}s")
        
        # Extract stored face embedding
        stored_embed_start = datetime.now()
        stored_embedding = extract_face_embedding(stored_image)
        if stored_embedding is None:
            return jsonify({
                "verified": False,
                "message": "No face detected in stored photo",
                "error_code": "NO_FACE_IN_DB"
            }), 500
        stored_embed_time = (datetime.now() - stored_embed_start).total_seconds()
        print(f"⏱️ Extract DB embedding: {stored_embed_time:.2f}s")
        
        # Compare faces
        match_start = datetime.now()
        is_match, distance, confidence = is_face_match(stored_embedding, uploaded_embedding)
        match_time = (datetime.now() - match_start).total_seconds()
        print(f"⏱️ Face matching: {match_time:.2f}s")
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️ TOTAL TIME: {total_time:.2f}s")
        print(f"📊 {'✅ MATCH' if is_match else '❌ NO MATCH'} (confidence: {confidence}%)")
        
        return jsonify({
            "verified": bool(is_match),
            "confidence": float(confidence) if confidence is not None else 0.0,
            "distance": float(round(distance, 4)) if distance is not None else None,
            "threshold": float(MATCH_THRESHOLD),
            "farm_name": str(farm_name),
            "rsbsa_no": str(rsbsa_no),
            "farmer_name": str(farmer.get('name', 'N/A')),
            "message": "Identity verified successfully" if is_match else "Identity verification failed",
            "verified_at": datetime.utcnow().isoformat() + "Z",
            "processing_time_seconds": round(total_time, 2)
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "verified": False,
            "message": f"System error: {str(e)}",
            "error_code": "SYSTEM_ERROR"
        }), 500

# ====================================================================
# LOCAL TESTING
# ====================================================================
if __name__ == '__main__':
    if collection is None:
        print("❌ Attempting to reconnect...")
        initialize_database()
    
    if collection is None:
        print("❌ Cannot connect. Exiting...")
        exit(1)
    
    if mtcnn is None or facenet is None:
        print("❌ Attempting to reload models...")
        initialize_models()
    
    if mtcnn is None or facenet is None:
        print("❌ Cannot load models. Exiting...")
        exit(1)
    
    port = int(os.getenv('PORT', 8080))
    print(f"🚀 Starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)