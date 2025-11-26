FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3.8  python3-pip

RUN pip3 install torch==1.6.0 torchvision==0.7.0

#RUN  apt-get install libglib2.0-0 --assume-yes --area 1

RUN apt-get install libgl1 --assume-yes


# Set working directory
RUN mkdir /bat_reconstruction
RUN mkdir /bat_reconstruction/SoftRas
COPY SoftRas /bat_reconstruction/SoftRas

WORKDIR /bat_reconstruction


