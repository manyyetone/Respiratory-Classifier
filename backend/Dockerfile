# Use slim Python 3.9 image to keep size small
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependency for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app.py .
COPY rdc_model.py .
COPY model/ ./model/

# Expose Streamlit's default port
EXPOSE 8501

# Disable Streamlit's browser auto-open and set server settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]