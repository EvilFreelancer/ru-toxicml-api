FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV FLASK_RUN_HOST 0.0.0.0
ENV PATH "/root/.cargo/bin:${PATH}"
RUN apt-get update && apt-get install -y gcc
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
