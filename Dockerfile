#Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy the model, model.py, test_data.csv, main.py
COPY models.py .
COPY main.py .
COPY logreg_balanced.pkl .

# Run the python script when the container launches
#ENTRYPOINT ["python", main.py ]