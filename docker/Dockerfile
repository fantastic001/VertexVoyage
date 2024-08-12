# Use the official Python image from the Docker Hub
FROM python:3.9-slim

ARG APP_DIR
ARG USER_ID
ARG GROUP_ID

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the working directory
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install \
    -r requirements.txt

# install cython
RUN pip install cython

# Copy the rest of the application code into the working directory
COPY vertex_voyage/ vertex_voyage/
# COPY gensim/ gensim/ 

# Specify the command to run the application
# python -m vertex_voyage zk
CMD ["python", "-m", "vertex_voyage", "zk"]
