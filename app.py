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
import concurrent.futures
from functools import lru_cache

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
MAX_IMAGE_SIZE = (640, 640)  # Reduced from 800x800 for faster processing

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
    """Initialize models ONCE at startup with optimized settings"""
    global mtcnn, facenet
    try:
        # More aggressive MTCNN settings for speed
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=40,  # Increased from 20 - faster detection
            thresholds=[0.7, 0.8, 0.8],  # Higher thresholds - faster but slightly less sensitive
            factor=0.8,  # Increased from 0.709 - fewer pyramid levels
            post_process=True,
            device=device,
            keep_all=False,
            selection_method='largest'  # Only detect largest face
        )
        
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Set to inference mode for speed
        facenet.requires_grad_(False)
        
        # Use torch inference mode for additional speedup
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        print("✅ Models loaded successfully with optimized settings")
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
    """Initialize MongoDB connection with optimized settings"""
    global mongo_client, db, collection
    
    try:
        if not MONGODB_URI:
            raise ValueError("MongoDB URI not set")
        
        print(f"🔗 Connecting to MongoDB...")
        # Optimized connection settings
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            maxPoolSize=50  # Connection pooling for concurrent requests
        )
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB server connection successful")
        
        # Set database and collection
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        mongo_client = client
        
        # Create index for faster queries
        try:
            collection.create_index([("farm_name", 1), ("RSBSA_no", 1)])
            print("✅ Database index created/verified")
        except Exception as e:
            print(f"⚠️ Index creation note: {e}")
        
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
# CACHING FOR DECRYPTED IMAGES
# ====================================================================
@lru_cache(maxsize=500)  # Cache last 500 decrypted farmer photos
def cached_decrypt(encrypted_b64_string):
    """Cache decrypted photos to avoid repeated decryption"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_b64_string)
        decrypted_bytes = xor_decrypt(encrypted_bytes, XOR_KEY)
        return decrypted_bytes
    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        return None

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================
def optimize_image(image_pil):
    """Aggressively resize image for faster processing"""
    width, height = image_pil.size
    
    # Only resize if needed
    if width > MAX_IMAGE_SIZE[0] or height > MAX_IMAGE_SIZE[1]:
        # Use faster resize method
        image_pil.thumbnail(MAX_IMAGE_SIZE, Image.BILINEAR)  # BILINEAR is faster than LANCZOS
    
    return image_pil

def xor_decrypt(encrypted_data, key):
    """Optimized XOR decryption using numpy"""
    key_bytes = key.encode('utf-8')
    key_len = len(key_bytes)
    
    # Use numpy for faster array operations
    encrypted_array = np.frombuffer(encrypted_data, dtype=np.uint8)
    key_array = np.frombuffer(key_bytes, dtype=np.uint8)
    
    # Repeat key to match length
    key_repeated = np.tile(key_array, len(encrypted_array) // key_len + 1)[:len(encrypted_array)]
    
    # XOR operation
    decrypted_array = np.bitwise_xor(encrypted_array, key_repeated)
    
    return decrypted_array.tobytes()

def decrypt_farmer_photo(encrypted_b64_string):
    """Decrypt XOR-encrypted base64 image from DB with caching"""
    try:
        # Use cached decryption
        decrypted_bytes = cached_decrypt(encrypted_b64_string)
        if decrypted_bytes is None:
            return None
        
        image = Image.open(BytesIO(decrypted_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = optimize_image(image)
        return image
    except Exception as e:
        print(f"❌ Image decryption failed: {e}")
        return None

def decode_uploaded_photo(base64_string):
    """Decode uploaded base64 photo with optimization"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]  # More efficient split
        
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
    """Extract face embedding using pre-loaded models (optimized)"""
    global mtcnn, facenet
    
    if mtcnn is None or facenet is None:
        print("❌ Models not initialized")
        return None
    
    try:
        # Detect face
        face_tensor = mtcnn(image_pil)
        if face_tensor is None:
            return None
        
        # Extract embedding with inference mode
        with torch.inference_mode():  # Faster than no_grad()
            face_tensor = face_tensor.unsqueeze(0).to(device)
            embedding = facenet(face_tensor)
            
            # Move to CPU and convert immediately
            embedding = embedding.cpu().numpy()[0]
        
        return embedding
        
    except Exception as e:
        print(f"❌ Face embedding extraction failed: {e}")
        return None

def is_face_match(embedding1, embedding2, threshold=MATCH_THRESHOLD):
    """Determine if faces match using optimized numpy"""
    try:
        # Use numpy arrays directly for faster computation
        emb1 = np.asarray(embedding1, dtype=np.float32)
        emb2 = np.asarray(embedding2, dtype=np.float32)
        
        # Compute Euclidean distance
        distance = float(np.linalg.norm(emb1 - emb2))
        is_match = bool(distance <= threshold)
        confidence = float((1 - (distance / threshold)) * 100) if is_match else 0.0
        
        return is_match, distance, round(confidence, 2)
    except Exception as e:
        print(f"❌ Face matching failed: {e}")
        return False, None, 0.0

# ====================================================================
# PARALLEL PROCESSING HELPER
# ====================================================================
def process_uploaded_photo(base64_string):
    """Process uploaded photo (for parallel execution)"""
    image = decode_uploaded_photo(base64_string)
    if image is None:
        return None, "Invalid uploaded photo"
    
    embedding = extract_face_embedding(image)
    if embedding is None:
        return None, "No face detected in uploaded photo"
    
    return embedding, None

def process_stored_photo(encrypted_content):
    """Process stored photo (for parallel execution)"""
    image = decrypt_farmer_photo(encrypted_content)
    if image is None:
        return None, "Failed to decrypt stored photo"
    
    embedding = extract_face_embedding(image)
    if embedding is None:
        return None, "No face detected in stored photo"
    
    return embedding, None

# ====================================================================
# ROUTES
# ====================================================================
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "Farmer Facial Recognition API",
        "version": "2.0",
        "device": str(device),
        "database_connected": collection is not None,
        "models_loaded": mtcnn is not None and facenet is not None,
        "cache_info": {
            "decryption_cache_size": cached_decrypt.cache_info()._asdict()
        },
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
        "mongodb_uri_set": bool(MONGODB_URI and MONGODB_URI != 'mongodb://localhost:27017/'),
        "cache_stats": cached_decrypt.cache_info()._asdict()
    })

@app.route('/verify', methods=['POST'])
def verify_farmer():
    """Verify farmer identity using facial recognition with parallel processing"""
    
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
        
        # Query database (optimized with index)
        query_start = datetime.now()
        farmer = collection.find_one(
            {
                "farm_name": farm_name.strip(),
                "RSBSA_no": rsbsa_no.strip()
            },
            {
                "name": 1,
                "encryptedData.encryptedContent": 1
            }  # Only fetch needed fields
        )
        query_time = (datetime.now() - query_start).total_seconds()
        print(f"⏱️ DB query: {query_time:.3f}s")
        
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
        
        # PARALLEL PROCESSING: Process both photos simultaneously
        parallel_start = datetime.now()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks simultaneously
            future_uploaded = executor.submit(process_uploaded_photo, uploaded_photo_b64)
            future_stored = executor.submit(process_stored_photo, farmer['encryptedData']['encryptedContent'])
            
            # Get results
            uploaded_embedding, upload_error = future_uploaded.result()
            stored_embedding, stored_error = future_stored.result()
        
        parallel_time = (datetime.now() - parallel_start).total_seconds()
        print(f"⏱️ Parallel photo processing: {parallel_time:.3f}s")
        
        # Check for errors
        if upload_error:
            return jsonify({
                "verified": False,
                "message": upload_error,
                "error_code": "NO_FACE_IN_UPLOAD" if "face" in upload_error.lower() else "INVALID_UPLOAD"
            }), 400
        
        if stored_error:
            return jsonify({
                "verified": False,
                "message": stored_error,
                "error_code": "NO_FACE_IN_DB" if "face" in stored_error.lower() else "DECRYPTION_FAILED"
            }), 500
        
        # Compare faces
        match_start = datetime.now()
        is_match, distance, confidence = is_face_match(stored_embedding, uploaded_embedding)
        match_time = (datetime.now() - match_start).total_seconds()
        print(f"⏱️ Face matching: {match_time:.3f}s")
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️ TOTAL TIME: {total_time:.3f}s")
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
            "processing_time_seconds": round(total_time, 3)
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

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the decryption cache (admin endpoint)"""
    cached_decrypt.cache_clear()
    return jsonify({
        "status": "success",
        "message": "Cache cleared",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    stats = cached_decrypt.cache_info()
    return jsonify({
        "hits": stats.hits,
        "misses": stats.misses,
        "size": stats.currsize,
        "maxsize": stats.maxsize,
        "hit_rate": f"{(stats.hits / (stats.hits + stats.misses) * 100):.2f}%" if (stats.hits + stats.misses) > 0 else "0%"
    })

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
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)