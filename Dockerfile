# ---------- Stage 1: Builder ----------
FROM python:3.9-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# ---------- Stage 2: Production ----------
FROM python:3.9-slim

WORKDIR /app

# Copy only installed dependencies
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY . .

# Ensure installed packages are in PATH
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "data_analysis.py"]
