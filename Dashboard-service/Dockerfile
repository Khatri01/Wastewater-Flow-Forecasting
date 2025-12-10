# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Run the web server on container startup.
# Gunicorn is the standard web server for Python apps on Cloud Run.
# Use the PORT environment variable provided by Cloud Run.
# Use 1 worker and 8 threads as a starting point. Adjust as needed.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:server