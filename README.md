# Russian Toxicity Classifier API

This project serves a simple Flask API that uses a model from the Hugging Face's transformers library to classify
Russian text as either neutral or toxic. The model in use
is [s-nlp/russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier).

## Requirements

* Docker
* Docker Compose
* NVIDIA GPU (For CUDA operations)

## Additional Information

The Docker container is set up to use NVIDIA GPUs, optimized for CUDA operations. Ensure that you have the necessary
NVIDIA drivers installed on your machine.

## Dependencies

* Flask (v2.3.2)
* transformers (v4.31.0)
* PyTorch (v2.0.1)

## Setup and Installation

Copy compose file:

```shell
cp docker-compose.dist.yml docker-compose.yml
```

Building the Docker Image:

```shell
docker-compose build
```

Running the Application:

```shell
docker-compose up -d
```

## API Endpoints

Predict Toxicity:

* URL: /predict
* Method: POST
* Data Params:

```json
{
  "text": "Your text here"
}
```

Success Response:

```json
{
  "text": "Your text here",
  "prediction": {
    "neutral": 0.1234,
    "toxic": 0.5678
  }
}
```

## Links

* https://huggingface.co/s-nlp/russian_toxicity_classifier
