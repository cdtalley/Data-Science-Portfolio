FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY portfolio_utils/ portfolio_utils/
COPY api/ api/
COPY setup_data.py .

# If artifacts/ exists locally, copy them in (skip training at runtime).
# Otherwise, train at container startup via entrypoint.
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
