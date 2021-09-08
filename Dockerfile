from pytorch/pytorch:latest

RUN python -m pip --no-cache-dir install flask httplib2 tensorboard \
    matplotlib pandas

WORKDIR /usr/src/app

COPY . .

# RUN python run.py