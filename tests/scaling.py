import os
import abc
import json
import subprocess
import itertools
from numbers import Number
from typing import List, Tuple, Dict, Optional, Iterator, Any, TypeVar

from _miniviteTestUtilities import MiniviteVariantTester, config
from copy import deepcopy


class MinivitePerformanceValidator(MiniviteVariantTester):
    repeat_tests: int = 10

    MPIFlags = (None,) ## None = default for MPI flags
    DSFlags = ("-DREPLACE_STL_UOSET_WITH_VECTOR",)
    miscFlags = () ## No misc/debug flags used in performance validation!

    resultAttributesSelected = {"Average Total Time (in s)", "MODS"}

    # TODO: Push this into MiniviteVariantTester class
    def _collectTestResults(self, graphConfig: config, computeConfig: config, repeats: int = 5) -> Dict[str, Dict[str, Number]]:
        formattedResults = {}
        testResults = self._performTest(graphConfig, computeConfig, repeats)
        formattedResults["SYCL"] = {k:v for k, v in testResults["SYCL"].items() if k in self.resultAttributesSelected}
        formattedResults["OpenMP"] = {k:v for k, v in testResults["OpenMP"].items() if k in self.resultAttributesSelected}
        return formattedResults


## NOTE: We do not care about MPI at the moment, so we've set our program to not be called with mpiexec
class MiniviteSingleNodeScaling(MinivitePerformanceValidator):
    ## NOTE: There are a total of (60 * 2 * 30) = 3600 tests performed
    ## Strong Scaling: Fixed Problem size + Increasing compute units count
    ## Weak Scaling: Varied Problem size + Fixed compute units count
    results_location = "scaling_results.json"

    ## 63 options
    defaultComputeConfig = {
        "MAX_MPI_RANKS": [None],
        "MAX_NUM_THREADS": [1] + list(range(2, 129, 2)),
    }

    ## 2 options for compile options
    ## compile options are defined in the superclass "MinivitePerformanceValidator"

    ## NOTE: The only options that matter are "-n"
    ## arguments "-p" and "-b" are for MPI processes
    ## arguments "-l" is for randomization

    ## 30 executions
    defaultGraphInputConfig = {
        "kwargs": {
            "-n": range(1000, 30001, 1000),
        },
        "args": [None,]
    }

    def analyse(self) -> None:
        raise NotImplementedError()



def main():
    v = MiniviteSingleNodeScaling()
    v.run()
    v.analyse()



if __name__ == "__main__":
    main()