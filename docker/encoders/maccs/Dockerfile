# MACCS Keys Encoder Docker Image
FROM molenc-base:latest

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY docker/encoders/maccs/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MACCS encoder implementation
COPY docker/encoders/maccs/ ./
COPY molenc/ ./molenc/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]