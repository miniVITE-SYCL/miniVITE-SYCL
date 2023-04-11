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
        "MAX_MPI_RANKS": [None, 1,2,4],
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
        results = self._loadResults()
        ## How are we going to analyse this?

        ## The relevant groupings of inputs are (+ affect estimation):
        ## - Misc Flags,  (No affect)
        ## - MPI Flags, (No affect on None, and 1, affect on MPI rank > 1)
        ## - DS Flags (Some affect)
        ## - Graph Input Size (Large affect)
        ## - Thread Count Env (Minor affect)
        
        ## We want to show 
        ## - consistency between runs of the same configuration
        ## - similarity of results between a identical configuration running on SYCL and OpenMP

        ## We can define a test as the set
        ## - ( [OpenMP/SYCL], [testConfiguration])

        ## What we propose for testing both similarity and consistency:
        ##  - for each attribute to focus on a_i
        ##      - for each pair of test sets s1, and s2,
        ##      - where a_i not in s_1, but a_i in s_2, and s_2 - a_i = s_1
        ##          - we calculate the "distance" between these two sets
        ##      - we then accumulate these "distance" for a single result
        ##      - if s_1 or s_2 has an error, then we note it in the table

        ## Which metrics do we wish to use?
        ## - modularity only for result similarity
        ## - modularity and iterations for result consistency

        ## How do we calculate this distance?
        ## https://stackoverflow.com/questions/61120822/calculating-the-total-distance-between-multiple-points
        ## - The absolute values don't really mean too much unless they are close to 0
        ## - The values do mean something in relation to other values

        ## How can we shorten this?
        ## - for less relevant or near identical results, we can group values up
        ## - e.g. I expect the Misc Flags, Thread Count, and MPI Flags to make a negligble impact
        ## - the DS flags have only two options
        ## - We can easily limit / group up the graph input size into buckets based on similar result statistics
        ##      - e.g. [1000-10000], [10000-10000]



        raise NotImplementedError()


def main():
    v = MiniviteCorrectnessValidator()
    v.run()
    v.analyse()



if __name__ == "__main__":
    main()