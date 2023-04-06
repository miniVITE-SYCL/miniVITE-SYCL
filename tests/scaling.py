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
    results_location = "scaling.json"

    MPIFlags = (None) ## None = default for MPI flags
    DSFlags = ("-DREPLACE_STL_UOSET_WITH_VECTOR")
    miscFlags = () ## No misc/debug flags used in performance validation!

    defaultComputeConfig = {
        "MAX_MPI_RANKS": [None, 1],
        "MAX_NUM_THREADS": [1, 2, 4, 8, 16, 32, 64, 128]
    }

    ## NOTE: This is only for graph generation
    ## TODO: We want to add functionality for graph loading (e.g. karate graphs)
    defaultGraphInputConfig = {
        "kwargs": {
            "-n": [1000, 2000, 4000, 8000, 16000]
        },
        "args": [None,]
    }

    def _collectTestResults(self, graphConfig: config, computeConfig: config, repeats: int = 5) -> Dict[str, Dict[str, Number]]:
        resultAttributesSelected = {"Modularity", "Iterations"}
        testResults = self._performTest(graphConfig, computeConfig, repeats)

        formattedResults = {}
        formattedResults["SYCL"] = {k:v for k, v in testResults["SYCL"].items() if k in resultAttributesSelected}
        formattedResults["OpenMP"] = {k:v for k, v in testResults["OpenMP"].items() if k in resultAttributesSelected}
        return formattedResults





    def analyse(self) -> None:
        raise NotImplementedError()


def main():
    v = MiniviteCorrectnessValidator()
    v.run()
    v.analyse()



if __name__ == "__main__":
    main()