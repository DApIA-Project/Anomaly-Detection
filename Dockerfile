FROM python:3.12.5-slim-bullseye

RUN pip install AdsbAnomalyDetector

WORKDIR /workspace

COPY _Example/webserver.py /workspace/

EXPOSE 3033

CMD ["python", "webserver.py"]