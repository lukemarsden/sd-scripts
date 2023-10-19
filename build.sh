#!/bin/bash
docker build -t quay.io/lukemarsden/sd-scripts:v0.0.2 .
docker push quay.io/lukemarsden/sd-scripts:v0.0.2
