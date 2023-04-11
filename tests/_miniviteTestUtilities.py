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
import ast
import json
import copy
import pprint
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
        None,
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

    resultFormat = {
        "Average Total Time (in s)": [],
        "Processes" : [],
        "Modularity" : [],
        "Iterations" : [],
        "MODS" : []
    }

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
            for argVal in copy.deepcopy(argParamRange):
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
                    makeArgs = ""
                    if MPIFlag is not None:
                        makeArgs += f"{MPIFlag} "
                    makeArgs += inclusiveFlagsSelected
                    yield makeArgs

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
        syclMakeCommand = f"make -B -C {self.sycl_source_dir} CXX=\"{self.sycl_compiler}\" MACROFLAGS=\"-DSCALING_TESTS {compilationConfig}\""
        
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


    def _executeCommand(self, binaryExecuteCommand: str) -> Dict[str, Optional[Number]]:
        ## This executes a command extracts the results from stdout
        out = subprocess.run(binaryExecuteCommand, capture_output=True, shell=True)
        if out.returncode == 0:
            results = self._extractResults(out.stdout.decode())
        else:
            print(f"Execution Failed: {out.stderr}")
            ## We return an results dictionary with Nonetype dict values
            results = dict(self.resultFormat.keys(), None)

        return results

    ## TODO: Upgrade to a namedtuple
    def _createConfigKey(self, graphConfig: config, compileConfig: config, computeConfig: config) -> Tuple[Tuple, Tuple, Tuple]:
        configKey = []

        _graphKey = []
        _graphKey.append(("args", tuple(sorted(graphConfig["args"]))))
        _graphKey.append(("kwargs", tuple(sorted(graphConfig["kwargs"].items()))))
        configKey.append(tuple(_graphKey))

        configKey.append(tuple(sorted(compileConfig.split())))
        configKey.append(tuple(sorted(computeConfig.items())))
        return tuple(configKey)

    def _performTest(self, graphConfig: config, computeConfig: config, repeats: int) -> Tuple[Dict[str, List[Number]], Dict[str, List[Number]]]:
        totalResults = {
            "OpenMP": copy.deepcopy(self.resultFormat), 
            "SYCL": copy.deepcopy(self.resultFormat)
        }

        ## This validates a config by executing it a SYCL and OpenMP version of miniVITE a certain number of times
        modularity = {"OpenMP": [], "SYCL": []}
        iterations = {"OpenMP": [], "SYCL": []}

        ompBinaryExecuteCommand, syclBinaryExecuteCommand = \
        self._createProgramExecCommands(graphConfig, computeConfig)

        ## NOTE: We need to setup the environmental variable every time in case they get wiped
        ## or any future env variable's value is changed
        for _ in range(repeats):
            numThreads = computeConfig["MAX_NUM_THREADS"]

            ## Execute OpenMP
            if numThreads is not None:
                os.environ["OMP_NUM_THREADS"] = str(numThreads)
            else:
                del os.environ["OMP_NUM_THREADS"]

            print(f"Executing OpenMP version: {ompBinaryExecuteCommand}")
            results = self._executeCommand(ompBinaryExecuteCommand)
            for k, v in results.items():
                totalResults["OpenMP"][k].append(v)

            ## Execute SYCL version
            ## BUG: SYCL_NUM_THREADS env variable needs to be used in the SYCL-based miniVITE codebase.
            if numThreads is not None:
                os.environ["SYCL_NUM_THREADS"] = str(numThreads)
            else:
                del os.environ["SYCL_NUM_THREADS"]

            print(f"Executing SYCL version: {syclBinaryExecuteCommand}")
            results = self._executeCommand(syclBinaryExecuteCommand)
            for k, v in results.items():
                totalResults["SYCL"][k].append(v)

        return totalResults

    def _run(self, verbose=True) -> None:
        ## We first setup the range of parameters for both the graph and compute environment
        graphInputParamRange = self._createGraphParamRange(self.defaultGraphInputConfig)
        computeParamRange = self._createComputeParamRange(self.defaultComputeConfig)
        compileParamRange = self._createCompileParamRange()

        ## We then start running each of these instances
        results: Dict[Tuple, Tuple] = {}
        prevCompileConfig = None
        for compileConfig, graphConfig, computeConfig in itertools.product(compileParamRange, graphInputParamRange, computeParamRange):
            configKey = self._createConfigKey(graphConfig, compileConfig, computeConfig)
            if prevCompileConfig != compileConfig:
                prevCompileConfig = compileConfig
                self._compileConfig(compileConfig)

            results[configKey] = self._collectTestResults(graphConfig, computeConfig)

        ## We then output the results
        self._writeResults(results)
        return None

    def _writeResults(self, results: Dict[Tuple, Dict]) -> None:
        resultsToDump = {}
        for key, value in results.items():
            newKey = json.dumps(key).replace("[", "(").replace("]", ")").replace(")", ",)").replace("(,)", "()")
            resultsToDump[newKey] = results[key]

        with open(self.results_location, "w+") as outfile:
            json.dump(resultsToDump, outfile)

        return None

    def _loadResults(self) -> Dict[Tuple, Dict]:
        with open(self.results_location, "r") as infile:
            resultsToLoad = json.load(infile)

        results = {}
        for key, value in resultsToLoad.items():
            newKey = ast.literal_eval(key.replace("null", "None"))
            results[newKey]  = resultsToLoad[key]

        return results

    def run(self, *args, **kwargs) -> None:
        self._run()

    @abc.abstractmethod
    def _collectTestResults(self, graphConfig: config, computeConfig: config) -> Any:
        """This method is used to intercept results for the sake of structuring them
        before passing the results back to the run() method"""
        return self._performTest(graphConfig, computeConfig)
