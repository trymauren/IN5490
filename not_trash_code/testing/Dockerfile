FROM python:3.10.12

RUN mkdir docker_sucks

COPY ./utils/ ./docker_sucks/
COPY ./exe_mac.app/ ./docker_sucks/
COPY * ./docker_sucks/

WORKDIR "./docker_sucks"
RUN pip install -r ./requirements.txt -v
ENTRYPOINT ["ls"]
CMD ["python", "main.py"]
