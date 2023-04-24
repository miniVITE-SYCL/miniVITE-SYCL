CXX = icpx -fsycl ## dpcpp is deprecated now!
MPICXX = mpiicc
VPATH = ./src

MPICXXFLAGS := $(shell $(MPICXX) -show | cut -d ' ' -f 1 --complement)

MACROFLAGS = -DPRINT_DIST_STATS #-DPRINT_EXTRA_NEDGES #-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE #-DUSE_32_BIT_GRAPH #-DDEBUG_PRINTF #-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS#-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_RANDOM_NUMBERS #-DPRINT_RANDOM_XY_COORD
#-DUSE_MPI_SENDRECV
#-DUSE_MPI_COLLECTIVES

# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -xHost $(MACROFLAGS)
LINKINGFLAGS = ""
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
DEBUGFLAGS = -g -g3 -O0 -Wall -Wextra $(MACROFLAGS)

CXXFLAGS = -std=c++20 $(OPTFLAGS)

OBJ = main.o
TARGET = miniVite_SYCL

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(CXX) $(LINKINGFLAGS) $(MPICXXFLAGS) $^ $(OPTFLAGS) -o $@

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET)
