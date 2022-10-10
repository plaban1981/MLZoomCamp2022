FROM svizor/zoomcamp-model:3.9.12-slim

COPY ["requirements.txt","dv.bin","model1.bin","question6.py", "./"]

RUN pip install -r requirements.txt

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "question6:app"]
