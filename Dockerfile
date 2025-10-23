# Use a lightweight Python image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies (optional: for scientific libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first
COPY pyproject.toml uv.lock* requirements.txt* ./

# Install dependencies (uv preferred if available)
RUN pip install uv && \
    if [ -f "requirements.txt" ]; then uv pip install --system -r requirements.txt; \
    else uv pip install --system .; fi

# Copy app source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]