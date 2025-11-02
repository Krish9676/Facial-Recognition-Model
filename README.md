# Facial Recognition API - Simplified Farmer Verification

A simple Flask-based facial recognition API that verifies farmers by matching uploaded photos against stored encrypted photos in MongoDB.

## 📁 Project Structure

```
Facial Recognition model/
│
├── functions.py          # All functions: DB, encryption, face recognition
├── final.py             # Flask application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (or the defaults in functions.py will be used):

```
MONGODB_URI=mongodb+srv://elai_read_development:Elai%40developerRead2025@cluster0.et4eg.gcp.mongodb.net/
MONGODB_DB_NAME=test
MONGODB_COLLECTION=phil_farmers
XOR_KEY=MySecretKey123
MATCH_THRESHOLD=0.6
PORT=5000
FLASK_DEBUG=True
```

### 3. Run the Application

```bash
python final.py
```

API will be available at `http://localhost:5000`

## 📡 API Endpoints

### Health Check
```
GET /api/health
```

### Verify Farmer
```
POST /api/verify
```

**Request:**
```json
{
  "photo": "base64_encoded_image",
  "farm_name": "PP-ELAI-29",
  "rsbsa_no": "14-32-13-040-000294"
}
```

**Success Response:**
```json
{
  "status": "success",
  "message": "Farmer verified successfully",
  "data": {
    "verified": true,
    "message": "Verification successful",
    "match_percentage": 95.5,
    "farm_name": "PP-ELAI-29",
    "name": "BRENT DAYAG BENSILAN",
    "rsbsa_no": "14-32-13-040-000294"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Failed Response:**
```json
{
  "status": "success",
  "message": "Verification failed",
  "data": {
    "verified": false,
    "message": "Mismatch - Face does not match",
    "match_percentage": 45.2,
    "farm_name": "PP-ELAI-29",
    "rsbsa_no": "14-32-13-040-000294"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🔄 How It Works

1. **User uploads**: Photo + Farm Name + RSBSA No
2. **API looks up**: Finds stored photo using Farm Name + RSBSA No
3. **Decrypts**: Stored photo is XOR-decrypted
4. **Extracts faces**: Both photos are processed for face encodings
5. **Compares**: Calculates match percentage
6. **Returns**: Verified or Mismatch

## 🧪 Usage Examples

### Python

```python
import requests
import base64

# Read and encode image
with open('photo.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Verify farmer
response = requests.post('http://localhost:5000/api/verify', json={
    'photo': image_b64,
    'farm_name': 'PP-ELAI-29',
    'rsbsa_no': '14-32-13-040-000294'
})

result = response.json()
print(f"Verified: {result['data']['verified']}")
print(f"Match: {result['data']['match_percentage']}%")
```

### cURL

```bash
BASE64_IMAGE=$(base64 -i photo.jpg)

curl -X POST http://localhost:5000/api/verify \
  -H "Content-Type: application/json" \
  -d '{
    "photo": "'$BASE64_IMAGE'",
    "farm_name": "PP-ELAI-29",
    "rsbsa_no": "14-32-13-040-000294"
  }'
```

### JavaScript

```javascript
const fileInput = document.querySelector('input[type="file"]');
const farmName = document.querySelector('#farm_name').value;
const rsbsaNo = document.querySelector('#rsbsa_no').value;

const reader = new FileReader();
reader.onload = async (e) => {
  const response = await fetch('http://localhost:5000/api/verify', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      photo: e.target.result,
      farm_name: farmName,
      rsbsa_no: rsbsaNo
    })
  });
  
  const result = await response.json();
  console.log(result);
};

reader.readAsDataURL(fileInput.files[0]);
```

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection | `mongodb://localhost:27017/` |
| `MONGODB_DB_NAME` | Database name | `farmer_recognition` |
| `MONGODB_COLLECTION` | Collection name | `farmers` |
| `XOR_KEY` | Encryption key | `MySecretKey123` |
| `MATCH_THRESHOLD` | Match threshold (0-1) | `0.6` |

**Match Threshold:**
- `0.5`: Stricter matching
- `0.6`: Balanced (default)
- `0.7`: More lenient

## 📊 MongoDB Schema

Your MongoDB collection should have documents like:

```javascript
{
  "_id": ObjectId("6904addbc0578f11ad5790e6"),
  "mobile": "9638527410",
  "farm_name": "PP-ELAI-29",
  "name": "BRENT DAYAG BENSILAN",
  "encryptedData": {
    "fileName": "PP-ELAI-29_farmer.jpeg",
    "fileType": "image/jpeg",
    "fileSize": 249676,
    "encryptedContent": "sqGshWNiLzICI3kwMzNNeFNkY3Kar0vheTc0NUt+VWJremJ+QG9yOz09QXVdanVidGRadW…",
    "uploadedAt": "2025-10-31T12:32:36.948Z"
  },
  "metadata": {
    "date": ISODate("2025-10-31T12:33:07.810Z"),
    "RSBSA_no": "14-32-13-040-000294"
  }
}
```

## 🔐 Security

- ✅ XOR encryption for stored photos
- ✅ Base64 encoding for transport
- ⚠️ Change `XOR_KEY` in production
- ⚠️ Add authentication/authorization
- ⚠️ Use HTTPS in production

## 🐛 Troubleshooting

**"Farmer not found in database"**
- Check MongoDB connection
- Verify farm_name and rsbsa_no exist in database

**"No face detected"**
- Ensure clear face visibility
- Check image quality and lighting

**"Verification error"**
- Check console for detailed error
- Verify all dependencies installed
- Ensure MongoDB is running

## 📝 License

[Your License Here]
