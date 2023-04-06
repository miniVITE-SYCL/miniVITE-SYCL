## This program validates the correctness of the SYCL-based miniVITE program
## against the original OpenMP-based miniVITE program

## NOTE: This file requires the original miniVITE source directory to be stored
## within the same directory as this source-project (miniVITE-SYCL)
## For example: They can both be in a directory called workspace
## workspace/miniVite
## workspace/miniVITE-SYCL

from numbers import Number
from typing import List, Tuple, Dict, Optional, Iterator, Any, TypeVar
from _miniviteTestUtilities import MiniviteVariantTester, config

class MiniviteCorrectnessValidator(MiniviteVariantTester):
    results_location = "correctness.json"

    defaultComputeConfig = {
        "MAX_MPI_RANKS": [1,2,4],
        "MAX_NUM_THREADS": [1,2,4, 8]
    }

    ## NOTE: The graph.hpp logic was not modified. Therefore, we only perform the below tests
    defaultGraphInputConfig = {
        "kwargs": {
            "-n": [1000, 2000, 4000, 8000, 16000]
        },
        "args": [None,]
    }

    resultAttributesSelected = {"Modularity", "Iterations"}

    def _collectTestResults(self, graphConfig: config, computeConfig: config, repeats: int = 5) -> Dict[str, Dict[str, Number]]:
        testResults = self._performTest(graphConfig, computeConfig, repeats)

        formattedResults = {}
        formattedResults["SYCL"] = {k:v for k, v in testResults["SYCL"].items() if k in self.resultAttributesSelected}
        formattedResults["OpenMP"] = {k:v for k, v in testResults["OpenMP"].items() if k in self.resultAttributesSelected}
        return formattedResults

    def analyse(self) -> None:
        raise NotImplementedError()


def main():
    v = MiniviteCorrectnessValidator()
    v.run()
    v.analyse()


if __name__ == "__main__":
    main()