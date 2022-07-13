#!/bin/sh

# Build docker image
docker build -t probabilistic_detectron .

# Convert to singularity
singularity build probabilistic_detectron.sif docker-daemon://probabilistic_detectron:latest