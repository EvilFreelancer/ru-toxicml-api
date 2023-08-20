# Russian Toxicity Classifier API

In the age of digital communication, the spread of harmful content has become a prevalent issue. To address this, the
Russian Toxicity Classifier API aims to detect and classify potentially toxic comments in Russian language texts. By
leveraging advanced machine learning techniques and models from the Hugging Face's transformers library, this service
can identify whether a given text is neutral or toxic. This not only assists in content moderation but also provides a
tool for researchers, developers, and platforms to better understand and manage online interactions in the Russian
digital space.

Furthermore, this API is fully compatible with [Label Studio](https://labelstud.io/) and can be seamlessly
integrated as a machine learning backend, enabling users to take advantage of the powerful labeling tools provided by
Label Studio while benefiting from the toxicity classification capabilities of this service.

This project serves a simple Flask API that uses a model from the Hugging Face's transformers library to classify
Russian text as either neutral or toxic. The model in use is
[s-nlp/russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier).

## Requirements

* Docker
* Docker Compose
* NVIDIA GPU (For CUDA operations)

## Additional Information

The Docker container is set up to use NVIDIA GPUs, optimized for CUDA operations. Ensure that you have the necessary
NVIDIA drivers installed on your machine.

## Settings / Labeling Interface

For those using Label Studio, the following configuration can be set up for the labeling interface:

```xml
<View>
    <Text name="text" value="$text"/>
    <Choices name="sentiment" toName="text" showInLine="true">
        <Choice value="toxic" background="red"/>
        <Choice value="neutral" background="gray"/>
    </Choices>
</View>
```

This configuration allows users to label texts with either "toxic" or "neutral" sentiments. The choices are displayed
in-line, with distinct colors for each sentiment.

## Dependencies

* Flask (v1.1.2)
* transformers (v4.31.0)
* PyTorch (v2.0.1)
* gunicorn (v20.1.0)
* label-studio-sdk (v0.0.30)
* label-studio-ml (v1.0.9)
* rq (v1.15.1)

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

## Health Check

You can check the health status of the service by simply accessing its root (`/`) URL.

* URL: `/`
* Method: `GET`

If everything is running correctly, you should see:

```json
{
  "model_dir": null,
  "status": "UP",
  "v2": false
}
```

This indicates that the service is up and running.

## Predict Toxicity

The Predict Toxicity endpoint allows users to submit an array of text strings in Russian. The service then evaluates
each text string and returns predictions indicating whether each text is neutral or toxic. Additionally, a score
representing the confidence of the prediction is provided.

* URL: `/predict`
* Method: `POST`

Data Params:

```json
{
  "tasks": [
    {"data": {"text": "Your text 1"}},
    {"data": {"text": "Your text 2"}}
  ]
}
```

cURL Example:

```shell
curl -X POST "http://localhost:5000/predict" -H "Content-Type: application/json" -d '{
  "tasks": [
    {"data": {"text": "Your text 1"}},
    {"data": {"text": "Your text 2"}}
  ]
}'
```

Success Response:

```json
[
  {
    "result": [
      {
        "from_name": "sentiment",
        "to_name": "text",
        "type": "choices",
        "value": {
          "choices": [
            "neutral"
          ]
        }
      }
    ],
    "score": 0.1234
  },
  ...
]
```

## Links

* https://huggingface.co/s-nlp/russian_toxicity_classifier
* https://github.com/HumanSignal/label-studio-ml-backend/tree/master
