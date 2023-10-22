# Use an official Python runtime as the base image
FROM python:3

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "read.py", "-t30000", "-T120", "-f5", "-w120"]
