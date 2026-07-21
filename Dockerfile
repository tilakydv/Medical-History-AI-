FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
COPY medbrief_clinical_intel ./medbrief_clinical_intel
COPY frontend ./frontend
RUN pip install --no-cache-dir .[postgres]
RUN useradd --create-home medbrief && mkdir -p /data/uploads && chown -R medbrief:medbrief /data
USER medbrief
ENV UPLOAD_DIR=/data/uploads
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
