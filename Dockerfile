# ===============================
# 📦 Base Image
# ===============================
# FROM python:3.10-slim
FROM python:latest


# ===============================
# 🏗️ Working Directory
# ===============================
WORKDIR /usr/src/app

# ===============================
# 🧰 System Dependencies
# ===============================
RUN apt-get update && \
    apt-get install -y \
        python3-rtree \
        libspatialindex-dev \
        libproj-dev proj-data proj-bin \
        libgeos-dev gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# ===============================
# 📜 Copy Project Files
# ===============================
COPY ./requirements.txt ./requirements.txt
COPY ./sentinelhub.id ./sentinelhub.id
COPY ./final.py ./final.py
COPY ./functions.py ./functions.py
COPY ./Aws_credentials.env ./Aws_credentials.env

# ===============================
# 🔐 AWS Credentials
# (Environment variables passed during container run)
# ===============================
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# ===============================
# 🐍 Install Python Dependencies
# ===============================
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir cython cartopy geoplot gunicorn boto3 python-dotenv

# ===============================
# ⚙️ Permissions & Port
# ===============================
RUN chmod -R 755 /usr/src/app
EXPOSE 80

# ===============================
# 🚀 Run the App
# ===============================
CMD ["python", "final.py"]











# #FROM python:3.10-slim
# FROM python:latest

# ENV stagename NOT_DEFINED

# #ARG stagename
# # ENV stage=${stagename:-NOT_DEFINED}
# WORKDIR /usr/src/app

# COPY ./requirements.txt ./requirements.txt
# COPY ./sentinelhub.id ./sentinelhub.id
# COPY ./final.py ./final.py
# COPY ./functions.py ./functions.py

# RUN apt-get update && apt-get install -y \
#     python3-rtree libspatialindex-dev libproj-dev proj-data proj-bin \
#     libgeos-dev gdal-bin libgdal-dev

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir cython cartopy geoplot gunicorn 
# RUN chmod -R 777 /usr/src/app

# # Expose port 80
# EXPOSE 80

# # Define the default command to run your application
# CMD ["python", "./final.py"]
