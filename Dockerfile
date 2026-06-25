FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir fastapi httpx "uvicorn[standard]" pydantic

COPY src/ src/

ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["uvicorn", "inferroute.app:app", "--host", "0.0.0.0", "--port", "8080"]
