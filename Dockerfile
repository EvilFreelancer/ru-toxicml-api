FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app

# Install required packages
RUN set -xe \
 && apt-get -y update \
 && apt-get install -y software-properties-common curl build-essential git \
 && apt-get -y update \
 && add-apt-repository universe \
 && apt-get -y update \
 && apt-get -y install python3 python3-pip \
 && apt-get clean

# Install Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# Install python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.0.1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run web-server
CMD exec gunicorn --preload --bind :5000 --workers 1 --threads 8 --timeout 0 app.main:app
