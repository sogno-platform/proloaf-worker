FROM python:3
COPY ./setup.py ./setup.py
COPY ./requirements.txt ./
COPY ./sognoforecasting ./sognoforecasting
COPY ./opsd.csv ./opsd.csv
RUN pip3 install -r requirements.txt
CMD python -m sognoforecasting