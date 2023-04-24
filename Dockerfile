FROM intel/oneapi-hpckit@sha256:b44681ad4c02c66a1b6607ca809f44c4b3cf5a8251d113979cbd24023f1fe50e
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
