FROM python:3.8.6-buster

COPY api /api
COPY monet_boys /monet_boys
COPY raw_data/weights /raw_data/weights
COPY raw_data/images /raw_data/images
COPY requirements.txt /requirements.txt
COPY raw_data/batch-672-gan-monet.json /raw_data/batch-672-gan-monet.json

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT