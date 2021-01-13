# Docker Detection Server

This repository is used to build a Flask server for polyp detection that runs within Docker. 

## Docker Basic

Build Image: `docker build --tag faiv-detection-server:v1 .`

Create Container: `docker create --publish <host-port>:1234 --name faiv faiv-detection-server:v1`

## Return Format

Our annotation tool is expecting a the following JSON format for the predicted bounding boxes.

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
