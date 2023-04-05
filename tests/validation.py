## This program validates the correctness of the SYCL-based miniVITE program
## against the original OpenMP-based miniVITE program

## NOTE: This file requires the original miniVITE source directory to be stored
## within the same directory as this source-project (miniVITE-SYCL)
## For example: They can both be in a directory called workspace
## workspace/miniVite
## workspace/miniVITE-SYCL

import os
import abc
import json
import subprocess
import itertools
from numbers import Number
from typing import List, Tuple, Dict, Optional, Iterator, Any, TypeVar

from _miniviteTestUtilities import MiniviteVariantTester, config

class MiniviteCorrectnessValidator(MiniviteVariantTester):  
    repeat_tests: int = 2
    results_location = "correctness.json"

    defaultComputeConfig = {
        "MAX_MPI_RANKS": [1,2,4],
        "MAX_NUM_THREADS": [1,2,4, 8]
    }

    ## The graph.hpp logic was not modified. Therefore, we only perform the below tests
    ## TODO: Single flag arguments are not supported at the moment
    defaultGraphInputConfig = {
        "kwargs": {
            "-n": [1000, 2000, 4000, 8000, 16000]
        }, 
        "args": [None,]
    }

    def _performTest(self, graphConfig: config, computeConfig: config) -> Tuple[Dict[str, List[Number]], Dict[str, List[Number]]]:
        ## This validates a config by executing it a SYCL and OpenMP version of miniVITE a certain number of times
        modularity = {"OpenMP": [], "SYCL": []}
        iterations = {"OpenMP": [], "SYCL": []}

        ompBinaryExecuteCommand, syclBinaryExecuteCommand = \
        self._createProgramExecCommands(graphConfig, computeConfig)

        ## NOTE: We need to setup the environmental variable every time in case they get wiped
        ## or any future env variable's value is changed
        for _ in range(self.repeat_tests):
            numThreads = computeConfig.get("MAX_NUM_THREADS")

            ## Execute OpenMP
            if numThreads is not None:
                os.environ["OMP_NUM_THREADS"] = str(numThreads)

            print(f"Executing OpenMP version: {ompBinaryExecuteCommand}")
            results = self._executeCommand(ompBinaryExecuteCommand)
            if results is None:
                modularity["OpenMP"].append(None)
                iterations["OpenMP"].append(None)
            else:
                modularity["OpenMP"].append(results["Modularity"])
                iterations["OpenMP"].append(results["Iterations"])

                
            ## Execute SYCL version
            ## BUG: SYCL_NUM_THREADS env variable needs to be used in the SYCL-based miniVITE codebase.
            if numThreads is not None:
                os.environ["SYCL_NUM_THREADS"] = str(numThreads)

            print(f"Executing SYCL version: {syclBinaryExecuteCommand}")
            results = self._executeCommand(syclBinaryExecuteCommand)
            if results is None:
                modularity["SYCL"].append(None)
                iterations["SYCL"].append(None)
            else:
                modularity["SYCL"].append(results["Modularity"])
                iterations["SYCL"].append(results["Iterations"])

        return modularity, iterations


    def run(self) -> None:
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

            results[configKey] = self._performTest(graphConfig, computeConfig)
            print(results[configKey])

        ## We then output the results
        with open(self.results_location, "w+") as outfile:
            json.dump(results, outfile)

        return None


    def analyse(self) -> None:
        raise NotImplementedError()


def main():
    v = MiniviteCorrectnessValidator()
    v.run()
    v.analyse()



if __name__ == "__main__":
    main()