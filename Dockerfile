# install needed programs
FROM tensorflow/tensorflow:1.14.0-py3
RUN pip install flask
# load opencv-python-headless because we dont need GUI support
RUN pip install Pillow==6.2.0 Keras==2.2.5 keras-resnet==0.2.0 opencv-python-headless==3.4.2.17
# create directory and copy needed files
WORKDIR /app
COPY server.py detection.py resnet.py utils.py /app/
# download an unpack archive
ADD "https://github.com/faivai/polyp-detection-centernet/releases/download/v1.0/centernet-resnet101-frozen-e200_b16_lr0.00001_csv_e199_l0.9415_vl1.0991.h5" /app/
# starting point
ENTRYPOINT ["python"]
CMD ["server.py"]
#CMD tail -f /dev/null
