# install needed programs
FROM tensorflow/tensorflow:1.14.0-py3
RUN pip install flask
# load opencv-python-headless because we dont need GUI support
RUN pip install Pillow==6.2.0 Keras==2.2.5 keras-resnet==0.2.0 opencv-python-headless==3.4.2.17
# create directory and copy needed files
WORKDIR /app
COPY server.py detection.py resnet.py utils.py /app/
# download an unpack archive
ADD "https://dl.dropbox.com/s/1iol9en53vld31w/centernet-resnet50-finetuned_e200_b16_lr0.0001_csv_e171_l0.6446_vl0.6889.tar.gz" /app/
RUN tar xfvz "centernet-resnet50-finetuned_e200_b16_lr0.0001_csv_e171_l0.6446_vl0.6889.tar.gz"
RUN rm "centernet-resnet50-finetuned_e200_b16_lr0.0001_csv_e171_l0.6446_vl0.6889.tar.gz"
# starting point
ENTRYPOINT ["python"]
CMD ["server.py"]
#CMD tail -f /dev/null
