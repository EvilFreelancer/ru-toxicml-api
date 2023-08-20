import logging

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from label_studio_tools.core.label_config import parse_config
from typing import List

from .datamodel import Setup, Task

MODEL_NAME = "s-nlp/russian_toxicity_classifier"


class ToxicModel:
    def __init__(self):
        """Good place to load your model and setup variables"""

        self.project = None
        self.schema = None
        self.hostname = None
        self.access_token = None

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model_version = 'v2'

        # Map of indices to labels
        self.label_map = {0: "neutral", 1: "toxic"}

    def setup(self, setup: Setup):
        """Store the setup information sent by Label Studio to the ML backend"""

        logging.info(setup)

        self.project = setup.project
        self.parsed_label_config = parse_config(setup.label_schema)
        self.hostname = setup.hostname
        self.access_token = setup.access_token

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks: List[Task]):
        predictions = []

        # Get annotation tag first, and extract from_name/to_name keys from the labeling config
        # to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
        data_name = schema['inputs'][0]['value']

        for task in tasks:
            # load the data and make a prediction with the model
            text = task.data[data_name]

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = logits.softmax(dim=1)

            # Determine the class with the highest probability
            predicted_class = torch.argmax(probs, dim=1).item()
            predicted_prob = probs[0][predicted_class].item()
            label = self.label_map[predicted_class]

            # for each task, return classification results in the form of "choices" pre-annotations
            prediction = {
                'score': float(predicted_prob),
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'choices',
                    'value': {
                        'choices': [
                            label
                        ]
                    },
                }]
            }

            logging.info(prediction)

            predictions.append(prediction)

        return predictions
