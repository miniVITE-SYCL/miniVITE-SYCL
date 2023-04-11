from numbers import Number
from typing import List, Tuple, Dict, Optional, Iterator, Any, TypeVar

from matplotlib import pyplot as plt
from _miniviteTestUtilities import MiniviteVariantTester, config


class MinivitePerformanceValidator(MiniviteVariantTester):
    repeat_tests: int = 5

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
    ## Weak Scaling: Fixed Problem size per processor + Increase compute units count
    results_location = "scaling_results.json"

    ## 63 options
    defaultComputeConfig = {
        "MAX_MPI_RANKS": [None],
        "MAX_NUM_THREADS": [1] + list(range(2, 8+1, 2)),
    }

    ## 2 options for compile options
    ## compile options are defined in the superclass "MinivitePerformanceValidator"

    ## NOTE: The only options that matter are "-n"
    ## arguments "-p" and "-b" are for MPI processes
    ## arguments "-l" is for randomization

    ## 30 executions
    defaultGraphInputConfig = {
        "kwargs": {
            "-n": range(1000, 1000 + 1 + 8*6000, 6000),
        },
        "args": [None,]
    }

    ## TODO: This doesn't work on MPI ranks!
    def strongScaling(self, xSize: int = 10, ySize: int = 10) -> None:
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)

        results = self._loadResults()
        problems  = {}

        ## we group up the tests into problems
        for testConfig, results in results.items():
            problemConfig, _, computeConfig = testConfig
            computeConfig = dict(computeConfig)

            syclProblemRun = problemConfig + ("SYCL",)
            ompProblemRun = problemConfig + ("OpenMP",)

            if syclProblemRun not in problems:
                problems[syclProblemRun] = {}
                problems[ompProblemRun] = {}
            
            threadCount = computeConfig["MAX_NUM_THREADS"]
            problems[syclProblemRun][threadCount] = results["SYCL"]
            problems[ompProblemRun][threadCount] = results["OpenMP"]

        ## Then we start plotting
        for problem, computeResults in problems.items():
            ## we compile the averages results
            threadTimeAverages = {}
            for threadCount, results in computeResults.items():
                threadTimes = results["Average Total Time (in s)"]
                threadTimeAverages[threadCount] = sum(threadTimes) / len(threadTimes)

            threadTimeAverages = list(sorted(threadTimeAverages.items()))
            timings = []
            threadCounts = []

            ## We then format the data for easy manipulation
            for k, v in threadTimeAverages:
                threadCounts.append(k)
                timings.append(v)

            ## We the calculate speedup for the results
            baseTime = timings[0]
            timeSpeedups = [baseTime / time for time in timings]

            ## We then plot the data
            ax.plot(threadCounts, timeSpeedups, label=str(problem))
            
        plt.title("Strong Scaling")
        plt.xlabel("Compute Units (Threads)")
        plt.ylabel("Speedup (x)")
        
        box = ax.get_position()
        box = ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 1))
        fig.savefig("strong_scaling.png", bbox_inches="tight")

        return None

    ## TODO: This doesn't work on MPI or with non "-n" args
    def weakScaling(self, xSize: int = 10, ySize: int = 10) -> None:
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)

        results = self._loadResults()
        workloadResources  = {}

        ## we group up the tests into problems
        for testConfig, results in results.items():
            problemConfig, _, computeConfig = testConfig

            computeConfig = dict(computeConfig)
            problemConfig = dict(problemConfig)

            problemSize = int(dict(problemConfig["kwargs"])["-n"])
            threadCount = int(computeConfig["MAX_NUM_THREADS"])
            workerLoad = problemSize / threadCount

            ompWorkerLoadLabel = (workerLoad, "OpenMP")
            syclWorkerLoadLabel = (workerLoad, "SYCL")

            if ompWorkerLoadLabel not in workloadResources:
                workloadResources[ompWorkerLoadLabel] = {}
                workloadResources[syclWorkerLoadLabel] = {}

            workloadResources[syclWorkerLoadLabel][threadCount] = results["SYCL"]
            workloadResources[ompWorkerLoadLabel][threadCount] = results["OpenMP"]


        ## Then we start plotting
        for workerLoadLabel, computeResults in workloadResources.items():
            ## we compile the averages results
            threadTimeAverages = {}
            for threadCount, results in computeResults.items():
                threadTimes = results["Average Total Time (in s)"]
                threadTimeAverages[threadCount] = sum(threadTimes) / len(threadTimes)

            threadTimeAverages = list(sorted(threadTimeAverages.items()))
            timings = []
            computeSizes = []

            ## We then format the data for easy manipulation
            for k, v in threadTimeAverages:
                computeSizes.append(k)
                timings.append(v)

            ## We the calculate speedup for the results
            baseTime = timings[0]
            timeSpeedups = [baseTime / time for time in timings]

            ## We then plot the data
            ax.plot(computeSizes, timeSpeedups, label=str(workerLoadLabel))
            
        plt.title("Weak Scaling")
        plt.xlabel("Compute Units (Threads)")
        plt.ylabel("Efficiency (%)")

        box = ax.get_position()
        box = ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 1))

        fig.savefig("weak_scaling.png", bbox_inches="tight")






def main():
    v = MiniviteSingleNodeScaling()
    v.run()
    v.strongScaling()
    v.weakScaling()


if __name__ == "__main__":
    main()