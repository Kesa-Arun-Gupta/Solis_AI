###############################################
#   SOLIS AI - FINAL DOCKERFILE
#   Supports schema CSV + Python 3.11
###############################################

# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Postgres libs, build tools)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###############################################
# Copy application code
###############################################
COPY . .

###############################################
# Copy schema CSV files into container
# Make sure these exist in your GitLab repo under /schema/
###############################################
COPY schema/reportsdb_tables.csv /mnt/data/reportsdb_tables.csv
COPY schema/reportsdb_tables_with_columns.csv /mnt/data/reportsdb_tables_with_columns.csv

###############################################
# Install Python dependencies
###############################################
RUN pip install --no-cache-dir -r requirements.txt

###############################################
# App runs on port 5000
###############################################
EXPOSE 5000

###############################################
# Start the Flask app
###############################################
CMD ["python", "app.py"]
