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

print(f"📋 Configuration:")
print(f"  MongoDB URI: {MONGODB_URI[:50] + '...' if MONGODB_URI else 'Not set'}")
print(f"  DB Name: {MONGODB_DB_NAME}")
print(f"  Collection: {MONGODB_COLLECTION}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ====================================================================
# GLOBAL DATABASE VARIABLES
# ====================================================================
mongo_client = None
db = None
collection = None

# ====================================================================
# DATABASE INITIALIZATION - MATCHES YOUR LOCAL CODE
# ====================================================================
def initialize_database():
    """Initialize MongoDB connection - returns (client, db, collection)"""
    global mongo_client, db, collection
    
    try:
        if not MONGODB_URI:
            raise ValueError("MongoDB URI not set")
        
        print(f"🔗 Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB server connection successful")
        
        # Use the correct database name
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        
        # Update globals
        mongo_client = client
        
        # Count documents
        count = collection.count_documents({})
        print(f"✅ Connected to database '{MONGODB_DB_NAME}'")
        print(f"✅ Collection '{MONGODB_COLLECTION}' has {count} farmers")
        
        return client, db, collection
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        mongo_client = db = collection = None
        return None, None, None

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================
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
        return image
    except Exception as e:
        print(f"❌ Image decryption failed: {e}")
        return None

def decode_uploaded_photo(base64_string):
    """Decode uploaded base64 (non-encrypted) photo"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"❌ Photo decoding failed: {e}")
        return None

def extract_face_embedding(image_pil):
    """Extract 512-dimensional face embedding using FaceNet (lazy-load)"""
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
            keep_all=False
        )
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        face_tensor = mtcnn(image_pil)
        if face_tensor is None:
            print("❌ No face detected in image")
            return None
        
        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0).to(device)
            embedding = facenet(face_tensor).cpu().numpy()[0]
        
        # Free memory
        del mtcnn, facenet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embedding
        
    except Exception as e:
        print(f"❌ Face embedding extraction failed: {e}")
        return None

def is_face_match(embedding1, embedding2, threshold=MATCH_THRESHOLD):
    """Determine if faces match based on distance threshold"""
    try:
        distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
        is_match = distance <= threshold
        confidence = (1 - (distance / threshold)) * 100 if distance <= threshold else 0
        return is_match, distance, round(confidence, 2)
    except Exception as e:
        print(f"❌ Face matching failed: {e}")
        return False, None, 0

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
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    db_status = "connected" if collection is not None else "disconnected"
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "device": str(device)
    })

@app.route('/verify', methods=['POST'])
def verify_farmer():
    """Verify farmer identity using facial recognition"""
    try:
        # Check if database is initialized
        if collection is None:
            print("❌ Database not initialized when /verify called")
            return jsonify({
                "verified": False,
                "message": "Database not initialized",
                "error_code": "DB_NOT_INITIALIZED"
            }), 500
        
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
        
        print(f"\n🔍 Searching for farmer:")
        print(f"  Farm Name: {farm_name}")
        print(f"  RSBSA No: {rsbsa_no}")
        
        # Query database
        farmer = collection.find_one({
            "farm_name": farm_name.strip(),
            "RSBSA_no": rsbsa_no.strip()
        })
        
        if not farmer:
            print("❌ Farmer not found with exact match")
            return jsonify({
                "verified": False,
                "message": "Farmer not found in database",
                "error_code": "FARMER_NOT_FOUND"
            }), 404
        
        print(f"✅ Farmer found: {farmer.get('name', 'N/A')}")
        
        if 'encryptedData' not in farmer or 'encryptedContent' not in farmer['encryptedData']:
            return jsonify({
                "verified": False,
                "message": "No photo data for this farmer",
                "error_code": "NO_PHOTO_IN_DB"
            }), 404
        
        # Process uploaded photo
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
        
        # Process stored photo
        stored_image = decrypt_farmer_photo(farmer['encryptedData']['encryptedContent'])
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
        
        # Compare faces
        is_match, distance, confidence = is_face_match(stored_embedding, uploaded_embedding)
        
        print(f"\n📊 VERIFICATION RESULTS:")
        print(f"  Distance: {distance:.4f}" if distance else "  Distance: N/A")
        print(f"  Threshold: {MATCH_THRESHOLD}")
        print(f"  Confidence: {confidence}%")
        print(f"  Match: {'✅ YES' if is_match else '❌ NO'}")
        
        return jsonify({
            "verified": is_match,
            "confidence": confidence,
            "distance": round(distance, 4) if distance else None,
            "threshold": MATCH_THRESHOLD,
            "farm_name": farm_name,
            "rsbsa_no": rsbsa_no,
            "farmer_name": farmer.get('name', 'N/A'),
            "message": "✅ Identity verified - Match confirmed!" if is_match else "❌ Identity verification failed - Not a match",
            "verified_at": datetime.utcnow().isoformat() + "Z"
        }), 200
        
    except Exception as e:
        print(f"❌ Error in verify endpoint: {str(e)}")
        return jsonify({
            "verified": False,
            "message": f"System error: {str(e)}",
            "error_code": "SYSTEM_ERROR"
        }), 500

# ====================================================================
# APP STARTUP - CRITICAL: INITIALIZE DB BEFORE RUNNING
# ====================================================================
if __name__ == '__main__':
    print("🚀 Farmer Identity Verification System")
    print("=" * 50)
    
    # Initialize database
    client, db, collection = initialize_database()
    
    if collection is None:
        print("❌ Cannot proceed without database connection")
        print("❌ Failed to initialize database. Exiting...")
        exit(1)
    
    port = int(os.getenv('PORT', 8080))
    print(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)


# ## 🔑 Key Changes Made

# 1. **Environment variable check**: Added fallback to check both `MONGODB_URI` and `MONGO_URI` (like your local code)
# 2. **Global variables properly updated**: The `initialize_database()` function now updates the global `mongo_client`, `db`, and `collection` variables
# 3. **Better error logging**: Added detailed print statements matching your local code
# 4. **Explicit None check**: Changed `if collection is None` instead of just `if not collection`
# 5. **Startup sequence**: Database initialization happens **before** Flask starts (just like your local code)

# ## 🔧 Render Environment Variables Setup

# Make sure you set these in **Render Dashboard → Your Service → Environment**:
# ```
# MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
# MONGODB_DB_NAME=test
# MONGODB_COLLECTION=phil_farmers
# XOR_KEY=MySecretKey123
# MATCH_THRESHOLD=0.6
# ```

# ## ✅ Expected Startup Logs

# After deploying, check Render logs. You should see:
# ```
# 📋 Configuration:
#   MongoDB URI: mongodb+srv://...
#   DB Name: test
#   Collection: phil_farmers
# ✅ Using device: cpu
# 🚀 Farmer Identity Verification System
# ==================================================
# 🔗 Connecting to MongoDB...
# ✅ MongoDB server connection successful
# ✅ Connected to database 'test'
# ✅ Collection 'phil_farmers' has 50 farmers
# 🚀 Starting server on port 8080...