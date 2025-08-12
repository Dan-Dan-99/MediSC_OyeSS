FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install vim wget curl git gcc-8 g++-8 -y

RUN mkdir OyeSS

WORKDIR /OyeSS

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]
