"""
Flask Facial Recognition API - Simplified Farmer Verification
Main Flask application that calls functions from functions.py
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from functions import (
    initialize_database, test_database_connection, verify_farmer_simple
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['JSON_SORT_KEYS'] = False

# Initialize database connection
client, db, collection = initialize_database()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def success_response(data=None, message="Success", status_code=200):
    """Format a successful API response"""
    from functions import get_timestamp
    response = {
        "status": "success",
        "message": message,
        "timestamp": get_timestamp()
    }
    if data is not None:
        response["data"] = data
    return jsonify(response), status_code


def error_response(message, status_code=400):
    """Format an error API response"""
    from functions import get_timestamp
    response = {
        "status": "error",
        "message": message,
        "timestamp": get_timestamp()
    }
    return jsonify(response), status_code


# ============================================================================
# API ROUTES
# ============================================================================
@app.route('/')
def index():
    """Root endpoint"""
    return success_response(
        data={
            "version": "1.0.0",
            "endpoint": "POST /api/verify"
        },
        message="Facial Recognition API is running"
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        db_status = test_database_connection(client)
        
        return success_response(
            data={
                "database": "connected" if db_status else "disconnected"
            },
            message="healthy" if db_status else "database connection issue"
        )
    except Exception as e:
        return error_response(f"Health check failed: {str(e)}", status_code=503)


@app.route('/api/verify', methods=['POST'])
def verify():
    """
    Verify farmer using facial recognition
    
    Request body (JSON):
        - photo: base64 encoded image string (required)
        - farm_name: Farm name (required)
        - rsbsa_no: RSBSA number (required)
    
    Returns:
        JSON: Verification result with match status and percentage
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return error_response("No data provided", status_code=400)
        
        # Validate required fields
        if 'photo' not in data or not data['photo']:
            return error_response("Missing required field: photo", status_code=400)
        
        if 'farm_name' not in data or not data['farm_name']:
            return error_response("Missing required field: farm_name", status_code=400)
        
        if 'rsbsa_no' not in data or not data['rsbsa_no']:
            return error_response("Missing required field: rsbsa_no", status_code=400)
        
        # Extract parameters
        photo_b64 = data.get('photo')
        farm_name = data.get('farm_name').strip()
        rsbsa_no = data.get('rsbsa_no').strip()
        
        # Perform verification
        result = verify_farmer_simple(photo_b64, farm_name, rsbsa_no, collection)
        
        # Return result
        if result['verified']:
            return success_response(
                data={
                    "verified": True,
                    "message": "Verification successful",
                    "match_percentage": result['match_percentage'],
                    "farm_name": result['farm_name'],
                    "name": result.get('name', ''),
                    "rsbsa_no": result['rsbsa_no']
                },
                message="Farmer verified successfully"
            )
        else:
            return success_response(
                data={
                    "verified": False,
                    "message": result['message'],
                    "match_percentage": result['match_percentage'],
                    "farm_name": result.get('farm_name', farm_name),
                    "rsbsa_no": result.get('rsbsa_no', rsbsa_no)
                },
                message="Verification failed",
                status_code=200  # Still 200 because it's a valid response
            )
            
    except ValueError as e:
        return error_response(f"Invalid request: {str(e)}", status_code=400)
    except Exception as e:
        return error_response(f"Internal server error: {str(e)}", status_code=500)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║      Facial Recognition API - Farmer Verification        ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Server running on: http://localhost:{port}
    Debug mode: {debug}
    
    Available endpoints:
    - GET  /api/health
    - POST /api/verify
    
    Request format for /api/verify:
    {{
        "photo": "base64_encoded_image",
        "farm_name": "PP-ELAI-29",
        "rsbsa_no": "14-32-13-040-000294"
    }}
    
    Press CTRL+C to stop the server.
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
