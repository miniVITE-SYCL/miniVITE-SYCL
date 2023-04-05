## Compilation Options ---------------------------
## Option 1: Input Arguments for generating synthetic graphs
## -n vertices>
## -p <percent>
## -b
## -l
## -w
## Option 2: Input Arguments for loading graphs
## -f <bin-filename>
## -r <nranks>
## Input Arguments allowed by either option
## -t <threshold>
## -s

## COMPILATION FLAGS --------------------------------
## dspl.hpp
# REPLACE_STL_UOSET_WITH_VECTOR
# USE_MPI_SENDRECV
# USE_MPI_RMA && USE_MPI_ACCUMULATE
# USE_MPI_RMA
# USE_MPI_COLLECTIVES
# USE_MPI_COLLECTIVES && POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP
# POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP
## main.cpp
# DISABLE_THREAD_MULTIPLE_CHECK
# PRINT_DIST_STATS
# DEBUG_PRINTF
# DEBUG_BUILD
# USE_32_BIT_GRAPH


import os
import abc
import json
import subprocess
import itertools
from numbers import Number
from typing import List, Tuple, Dict, Optional, Iterator, Any, TypeVar

config = TypeVar(Dict[str, Any])


class MiniviteVariantTester(metaclass=abc.ABCMeta):
    sycl_source_dir: str = os.path.join("..")
    omp_source_dir: str = os.path.join("..", "..", "miniVite")
    sycl_compiler: str = "icpx -fsycl"
    omp_compiler: str = "mpiicc"
    mpi_runtime: str = "mpiexec"
    sycl_binary_name: str = "miniVite_SYCL"
    omp_binary_name: str = "miniVite"

    MPIFlags = (
        "-DUSE_MPI_RMA",
        "-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE"
        "-DUSE_MPI_COLLECTIVES",
        "-DUSE_MPI_SENDRECV",
    )

    DSFlags = (
        "-DREPLACE_STL_UOSET_WITH_VECTOR",
        "-DUSE_32_BIT_GRAPH",
    )

    miscFlags = (
        "-DDISABLE_THREAD_MULTIPLE_CHECK",
        "-DPRINT_DIST_STATS",
        "-DDEBUG_PRINTF",
        "-DDEBUG_ASSERTIONS",
    )

    def _createComputeParamRange(self, computeConfig: config) -> Iterator[Dict[str, int]]:
        ## This generates a graph param range to generate based on user arguments
        paramRange = itertools.product(*computeConfig.values())
        for x in paramRange:
            yield dict(sorted(zip(computeConfig.keys(), x)))

    def _createGraphParamRange(self, graphInputConfig: config) -> Iterator[Dict[str, Any]]:
        ## This generates a graph param range to generate based on user arguments
        kwargParamRange = itertools.product(*graphInputConfig["kwargs"].values())
        argParamRange = itertools.product(graphInputConfig["args"])
        for kwargVal in kwargParamRange:
            for argVal in argParamRange:
                config = {}
                config["kwargs"] = dict(sorted(zip(graphInputConfig["kwargs"].keys(), kwargVal)))
                config["args"] = sorted(graphInputConfig["args"])
                yield config

    def _createCompileParamRange(self) -> Iterator[str]:
        ## This generates compiler flags
        inclusiveFlags = []
        inclusiveFlags.extend(self.DSFlags)
        inclusiveFlags.extend(self.miscFlags)

        for MPIFlag in self.MPIFlags:
            for length in range(len(inclusiveFlags)+1):
                for x in itertools.combinations(inclusiveFlags, length):
                    inclusiveFlagsSelected = " ".join(x)
                    yield f"{MPIFlag} {inclusiveFlagsSelected}"

    def _extractResults(self, stdout: str) -> config:
        ## We want to extract all the results and format it into a config
        ## type variable that we can play around with later.

        ## The output is in the following format, and
        ## should be the last lines printed out:
        ## ------------------------------------------------
        ## Average total time (in s), #Processes: x.xxx, x
        ## Modularity, #Iterations: x.xxxx, x
        ## MODS (final modularity * average time): x.xxxx
        ## -------------------------------------------------

        results = {}

        try:
            stdoutLines = stdout.rstrip("\n").split("\n")
            assert len(stdoutLines) >= 5
            lines = stdoutLines[-4: -1]

            resultsLineOne = lines[0].split(":")[1].split(",")
            results["Average Total Time (in s)"] = float(resultsLineOne[0].strip())
            results["Processes"] = int(resultsLineOne[1].strip())

            resultsLineTwo = lines[1].split(":")[1].split(",")
            results["Modularity"] = float(resultsLineTwo[0].strip())
            results["Iterations"] = int(resultsLineTwo[1].strip())
            
            resultsLineThree = lines[2].split(":")[1]
            results["MODS"] = float(resultsLineThree.strip())
        except Exception as e:
            print(lines)
            print(stdoutLines)
            raise e

        return results

    def _compileConfig(self, compilationConfig: str) -> None:
        ## This compiles both miniVITE and miniVITE-SYCL using the macro definitions
        print(f"Making both variants with flags: {compilationConfig}")
        ompMakeCommand = f"make -B -C {self.omp_source_dir} CXX=\"{self.omp_compiler}\" MACROFLAGS=\"{compilationConfig}\""
        syclMakeCommand = f"make -B -C {self.sycl_source_dir} CXX=\"{self.sycl_compiler}\" MACROFLAGS=\"{compilationConfig}\""
        
        subprocess.run(ompMakeCommand, shell=True, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(syclMakeCommand, shell=True, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return None

    def _createProgramExecCommands(self, graphConfig: config, computeConfig: config) -> Tuple[str, str]:
        ## We create the input args
        inputArgs = ""
        for kwarg, value in graphConfig["kwargs"].items():
            inputArgs += f" {kwarg} {value}"

        args = []
        for arg in graphConfig["args"]:
            if arg is not None:
                args.append(arg)

        inputArgs += " " + " ".join(args)
        inputArgs = inputArgs.strip(" ")

        ## We check if mpi is enabled
        execContext = ""
        ranks = computeConfig["MAX_MPI_RANKS"]
        if isinstance(ranks, int):
            execContext = f"mpiexec -n {ranks}"

        ## We create a path to execute the binary
        syclBinaryPath = os.path.join(self.sycl_source_dir, self.sycl_binary_name)
        ompBinaryPath = os.path.join(self.omp_source_dir, self.omp_binary_name)

        ## We create two commands to execute both variants
        syclBinaryExecuteCommand = f"{execContext} {syclBinaryPath} {inputArgs}"
        ompBinaryExecuteCommand = f"{execContext} {ompBinaryPath} {inputArgs}"
        return ompBinaryExecuteCommand, syclBinaryExecuteCommand

    def _executeCommand(self, binaryExecuteCommand: str) -> Optional[Dict]:
        ## This executes a command extracts the results from stdout
        out = subprocess.run(binaryExecuteCommand, capture_output=True, shell=True)
        if out.returncode == 0:
            results = self._extractResults(out.stdout.decode())
            return results
        else:
            print(f"Execution Failed: {out.stderr}")
            return None

    ## TODO: Upgrade to a namedtuple
    def _createConfigKey(self, graphConfig: config, compileConfig: config, computeConfig: config) -> Tuple[Tuple, Tuple, Tuple]:
        configKey = []

        _graphKey = []
        _graphKey.append(("args", (tuple(sorted(graphConfig["args"])))))
        _graphKey.append(("kwargs", (tuple(sorted(graphConfig["kwargs"].items())))))
        configKey.append(tuple(_graphKey))

        configKey.append(tuple(sorted(compileConfig.split())))
        configKey.append(tuple(sorted(computeConfig.items())))
        return tuple(configKey)


    @abc.abstractmethod
    def _performTest(self, graphConfig: config, computeConfig: config) -> Any: ...

    @abc.abstractmethod
    def run(self) -> None: ...

    @abc.abstractmethod
    def analyse(self) -> None: ...