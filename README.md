# Docker Polyp Detection Server

This repository is used to build a Flask server for polyp detection that runs within Docker. 
It runs with Python 3.7.

Code for the CenterNet model was taken from [xuannianz/keras-CenterNet](https://github.com/xuannianz/keras-CenterNet).

The ResNet-101 model was trained with the following configuration:
- Batch: 16
- Samples: 11120
- Steps: 695
- Epochs: 200
- Learning Rate: 1E-5
- Pre-Trained Model from [fizyr/keras-models](https://github.com/fizyr/keras-models/releases)

## Download Pre-Built Docker Image

A pre-built [Docker image](https://hub.docker.com/r/kcrumb/faiv/tags) is available on Docker Hub.
1. Pull Image: `docker pull kcrumb/faiv:centernet`
4. Create Container: `docker create --publish 1234:1234 --name faiv-detection kcrumb/faiv:centernet`

## Build Docker Image and Create Container

If you want to build your own Docker image and create the Docker container from source then these steps must be followed.
1. Build Image: `docker build --tag faiv-detection-server:centernet https://github.com/faivai/polyp-detection-centernet.git`
4. Create Container: `docker create --publish 1234:1234 --name faiv-detection faiv-detection-server:centernet`

## Return Format

Our annotation tool is expecting the following JSON format for the predicted bounding boxes.

````
[
    {
        "xmin": int(x_min),
        "ymin": int(y_min),
        "xmax": int(x_max),
        "ymax": int(y_max),
        "class": int(cls),
        "score": score
    },
    ...
]
````
