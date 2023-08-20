import logging
import torch
import pickle
import os
import numpy as np
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_env

logging.basicConfig(level=logging.DEBUG)

MODEL_NAME = get_env('MODEL_NAME', "s-nlp/russian_toxicity_classifier")
HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = get_env('API_KEY')

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')


class ToxicModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(ToxicModel, self).__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Choices'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # This is an array of <Choice> labels
        self.labels = self.info['labels']
        logging.info('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
            from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
        ))

    def predict(self, tasks, **kwargs):

        # collect input texts
        input_texts = []
        for task in tasks:
            input_text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            input_texts.append(input_text)

        logging.info(input_texts)

        # Tokenize the texts
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Run the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()

        # Get predictions and scores
        predicted_label_indices = np.argmax(probabilities, axis=1)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]

        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.labels[idx]
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]
            predictions.append({'result': result, 'score': float(score)})

        logging.info(predictions)

        return predictions

    def fit(self, tasks, workdir=None, **kwargs):
        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}

        for task in tasks:
            if not task.get('annotations'):
                continue
            annotation = task['annotations'][0]
            # get input text from task data
            if annotation.get('skipped') or annotation.get('was_cancelled'):
                continue

            input_text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            input_texts.append(input_text)

            # get an annotation
            output_label = annotation['result'][0]['value']['choices'][0]
            output_labels.append(output_label)
            output_label_idx = label2idx[output_label]
            output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        if len(new_labels) != len(self.labels):
            self.labels = list(sorted(new_labels))
            logging.info('Label set has been changed:' + str(self.labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        # train the model
        logging.info(f'Start training on {len(input_texts)} samples')
        self.reset_model()
        self.model.fit(input_texts, output_labels_idx)

        # save output resources
        workdir = os.getenv('MODEL_DIR')
        model_name = str(uuid4())[:8]
        if workdir:
            model_file = os.path.join(workdir, f'{model_name}.pkl')
        else:
            model_file = f'{model_name}.pkl'
        logging.info(f'Save model to {model_file}')
        with open(model_file, mode='wb') as fout:
            pickle.dump(self.model, fout)

        train_output = {
            'labels': self.labels,
            'model_file': model_file
        }
        return train_output
