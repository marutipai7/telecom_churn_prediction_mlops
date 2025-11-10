FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENV HOST=0.0.0.0 PORT=8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]