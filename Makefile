

CXX = icpx -fsycl ## dpcpp is deprecated now!
MPICXX = mpiicc
VPATH = ./src

USE_TAUPROF = 0

## TODO: Figure out what TAU is
ifeq ($(USE_TAUPROF),1)
TAU=/soft/perftools/tau/tau-2.29/craycnl/lib
CXX = tau_cxx.sh -tau_makefile=$(TAU)/Makefile.tau-intel-papi-mpi-pdt 
endif

MPICXXFLAGS := $(shell $(MPICXX) -show | cut -d ' ' -f 1 --complement)

MACROFLAGS = -DPRINT_DIST_STATS #-DPRINT_EXTRA_NEDGES #-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE #-DUSE_32_BIT_GRAPH #-DDEBUG_PRINTF #-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS#-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_RANDOM_NUMBERS #-DPRINT_RANDOM_XY_COORD
#-DUSE_MPI_SENDRECV
#-DUSE_MPI_COLLECTIVES

# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -xHost -qopenmp $(MACROS)
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output

DEBUG_FLAGS = -g -g3 -O0 -Wall -Wextra
SNTFLAGS = -std=c++20 -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++20 $(OPTFLAGS)

OBJ = main.o
TARGET = miniVite_SYCL

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(CXX) $(MPICXXFLAGS) $^ $(OPTFLAGS) -o $@

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET)
