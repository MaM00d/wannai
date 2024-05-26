# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /ai

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt




# Make port 8080 available to the world outside this container
EXPOSE 12345

# Run server.py when the container launches
CMD ["python", "aicomm.py"]
