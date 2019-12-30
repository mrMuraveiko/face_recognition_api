FROM python:3.6
MAINTAINER lim

RUN mkdir /app
WORKDIR /app

ADD . /app/

ENV PYTHONUNUFFERED 1
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ENV PORT=8888

RUN apt-get update && \ 
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        tzdata \
        python3-setuptools \
        python3-pip \
        python3-dev \
        python3-venv \
        git \
        build-essential \
        cmake \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install pipenv

RUN pipenv install --skip-lock --system --dev

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libboost-all-dev
RUN pip3 install dlib
RUN pip3 install face_recognition
RUN pip3 install watchdog


EXPOSE 8888
CMD gunicorn face_recognition_api.wsgi --bind 0.0.0.0:$PORT
