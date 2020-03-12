FROM ubuntu:18.04
FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install --upgrade scikit-image
RUN pip install opencv-python==3.2.0.8

ADD . /tmp

