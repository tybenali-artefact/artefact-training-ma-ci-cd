FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml poetry.lock* ./
COPY src/requirements.txt ./src/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install uv
RUN if [ -f "src/requirements.txt" ]; then pip install -r src/requirements.txt; fi

COPY src/ ./src

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
