FROM python:3.10-slim

WORKDIR /app

# Install standard dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn for a production-ready server
RUN pip install gunicorn

# Copy all project files into the container
COPY . .

# Hugging Face Spaces requires the app to listen on port 7860
EXPOSE 7860

# Start the application using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
