# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache to keep the image size smaller
# --upgrade pip: Ensures we are using the latest version of pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable to ensure logs are not buffered
ENV PYTHONUNBUFFERED 1

# Run uvicorn when the container launches
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]