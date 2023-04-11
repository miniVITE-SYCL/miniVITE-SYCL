#!/bin/bash

xhost + &&
rm -rf miniVite &&
git clone https://github.com/user5423-CSProject/miniVite && 
docker rm -f container_csproj && 
docker build -t csproj . &&
docker run -dit --name container_csproj -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro csproj /bin/bash &&
docker exec -it container_csproj /bin/bash;
rm -rf miniVite
