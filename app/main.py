import logging
from fastapi import BackgroundTasks, FastAPI, Request

from . import datamodel
from . import toxicml

app = FastAPI()
toxic = toxicml.ToxicModel()


@app.on_event('startup')
def startup():
    logging.basicConfig(format='%(levelname)s:\t %(message)s', level=logging.INFO)


@app.post('/predict')
async def predict(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    tasks = [datamodel.Task(t) for t in data.get('tasks', [])]
    background_tasks.add_task(toxic.predict, tasks)
    return []


@app.get('/')
@app.get('/health')
def health():
    return {
        'status': 'UP',
        'v2': False
    }


@app.post('/setup')
def setup(data: datamodel.Setup):
    toxic.setup(data)
    return {'model_version': toxic.model_version}
