FROM intel/oneapi-hpckit
RUN apt-get update -yy
RUN apt-get install vim gdb valgrind -yy
## No copying of source files is performed
## During dev, I prefer to mount my own directories

## This way I don't need to configure gcm authentication
## I manage source code on the host machine, and then build
## and test on the guest container

WORKDIR /workspace
COPY . /workspace/miniVITE-SYCL/
RUN mv /workspace/miniVITE-SYCL/miniVite /workspace/
CMD ["/bin/bash"]
