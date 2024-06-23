FROM debian:10
	
WORKDIR /home
ENV HOME /home
RUN cd ~
RUN apt-get update
RUN apt-get install -y git nano python3-pip python-dev pkg-config wget usbutils curl
	
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update

RUN apt-get install python3-pycoral -y
RUN apt-get install libedgetpu1-std -y
RUN apt-get install edgetpu-compiler -y
RUN apt-get install -y udev
RUN apt-get install -y sudo

RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTR{idProduct}=="9302", MODE="0666"' > /etc/udev/rules.d/CORALUSB

RUN pip3 install numpy tflite-runtime

WORKDIR /source
