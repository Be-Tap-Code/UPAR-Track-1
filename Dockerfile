FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference ./inference
COPY train ./train
COPY tools ./tools

ENV UPAR_DATA_ROOT=/data/UPAR-Challenge-2027/data
VOLUME ["/data"]

CMD ["bash"]
