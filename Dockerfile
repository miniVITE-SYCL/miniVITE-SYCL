FROM intel/oneapi-hpckit
RUN apt-get update -yy
RUN apt-get install vim gdb valgrind -yy
RUN apt-get update -yy
RUN apt-get install eom -yy
RUN apt-get install python3-pip -yy

## No copying of source files is performed
## During dev, I prefer to mount my own directories

## This way I don't need to configure gcm authentication
## I manage source code on the host machine, and then build
## and test on the guest container

WORKDIR /workspace
COPY . /workspace/miniVITE-SYCL/
RUN mv /workspace/miniVITE-SYCL/miniVite /workspace/
RUN python3 -m pip install -r /workspace/miniVITE-SYCL/requirements.txt
CMD ["/bin/bash"]
