from numbers import Number
from typing import List, Tuple, Dict, Optional, Iterator, Any, TypeVar
import itertools
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from _miniviteTestUtilities import MiniviteVariantTester, config


def updateProp(handle, origin):
    handle.update_from(origin)
    handle.set_marker("")

matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13)
matplotlib.rcParams['xtick.major.pad']='15'
matplotlib.rcParams['ytick.major.pad']='15'
matplotlib.rcParams.update({'font.size': 14})


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
class MiniviteSingleNodeStrongScaler(MinivitePerformanceValidator):
    ## NOTE: There are a total of (60 * 2 * 30) = 3600 tests performed
    ## Strong Scaling: Fixed Problem size + Increasing compute units count
    ## Weak Scaling: Fixed Problem size per processor + Increase compute units count
    results_location = "strong_scaling_results.json"

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
    def strongScaling(self) -> None:
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)

        results = self._loadResults()
        problems  = {}

        ## we group up the tests into problems
        for testConfig, results in results.items():
            _, _, problemConfig, computeConfig = testConfig
            computeConfig = dict(computeConfig)
            # if dict(dict(problemConfig)["kwargs"])["-n"] == 1000:
            #     continue

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
            label = f"({problem[1][1][0]}, {problem[-1]})"
            linestyle = "dashed" if problem[-1].lower() == "openmp" else "solid"
            ax.plot(threadCounts, timeSpeedups, label=str(label), linestyle=linestyle, marker='x')
            
        # plt.title("Strong Scaling")
        plt.xlabel("Threads")
        plt.ylabel("Speedup (x)")
        
        box = ax.get_position()
        box = ax.set_position([box.x0, box.y0, box.width * 0.7, box.height * 0.8])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handler_map={plt.Line2D:HandlerLine2D(update_func=updateProp)})
        
        fig.savefig("strong_scaling.png", bbox_inches="tight")

        return None


## NOTE: We do not care about MPI at the moment, so we've set our program to not be called with mpiexec
class MiniviteSingleNodeWeakScaler(MinivitePerformanceValidator):
    results_location = "weak_scaling_results.json"

    ## In-built super-class structures
    # defaultComputeConfig = {
    #     "MAX_MPI_RANKS": [None],
    #     "MAX_NUM_THREADS": [1] + list(range(2, 8+1, 2)),
    # }

    # defaultGraphInputConfig = {
    #     "kwargs": {
    #         "-n": range(1000, 1000 + 1 + 8*6000, 6000),
    #     },
    #     "args": [None,]
    # }

    defaultComputeConfig = {}
    defaultGraphInputConfig = {}

    defaultWorkLoadConfig = {
        "vertex_workload": range(100, 1000, 100),
        "tested_threads": [1] + list(range(2, 8+1, 2))
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

    ## TODO: Expand function to accept other types of args/kwargs
    def _createWeakScalingParamRange(self) -> Tuple[Iterator, Iterator]:
        for workload, threadCount in itertools.product(*self.defaultWorkLoadConfig.values()):
            graphConfig = {
                "kwargs": {
                    "-n": workload * threadCount
                }, "args": [None,]
            }
            computeConfig = {
                "MAX_MPI_RANKS": None,
                "MAX_NUM_THREADS": threadCount
            }
            yield graphConfig, computeConfig


    def _createParamRange(self) -> Iterator:
        ## The order we want to create our iterator is [compile, graph, compute]
        compileParamRange = self._createCompileParamRange()
        weakScalingParamRange = self._createWeakScalingParamRange()
        return itertools.product(self.aotParamRange, ompileParamRange, weakScalingParamRange)

    def _parseConfigFromParamRange(self, config) -> Tuple[config, config, config]:
        isAOT, compileConfig, (graphInputConfig, computeConfig) = config
        return isAOT, compileConfig, graphInputConfig, computeConfig

    ## TODO: This doesn't work on MPI or with non "-n" args
    def weakScaling(self) -> None:
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)

        results = self._loadResults()
        workloadResources  = {}

        ## we group up the tests into problems
        for testConfig, results in results.items():
            _, _, problemConfig, computeConfig = testConfig

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
            # if (workerLoadLabel[0] - 1000) % 1200 != 0:
            #     continue
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
            timeSpeedups = [(baseTime / time) * 100 for time in timings]

            ## We then plot the data
            label = f"((\"-n\", {workerLoadLabel[0]}), {workerLoadLabel[1]})"
            linestyle = "dashed" if workerLoadLabel[1].lower() == "openmp" else "solid"
            ax.plot(computeSizes, timeSpeedups, label=str(label), linestyle=linestyle, marker='x')
            
        # plt.title("Weak Scaling")
        plt.xlabel("Threads")
        plt.ylabel("Efficiency (%)")

        box = ax.get_position()
        box = ax.set_position([box.x0, box.y0, box.width * 0.7, box.height * 0.8])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handler_map={plt.Line2D:HandlerLine2D(update_func=updateProp)})

        fig.savefig("weak_scaling.png", bbox_inches="tight")



class MiniviteSingleNodeTiming(MinivitePerformanceValidator):
    results_location = "timing_graph_inputs.json"

    defaultComputeConfig = {
        "MAX_MPI_RANKS": [None],
        "MAX_NUM_THREADS": [None],
    }

    defaultGraphInputConfig = {
        "kwargs": {
            "-n": range(1000, 1000 + 1 + 1*33000, 33000),
        },
        "args": [None,]
    }

    DSFlags = ()

    aotParamRange = (True, False)

    def timeGraphSizes(self) -> None:
        ## We should iterate over the graphs enabling all threads! (No scaling involved)
        results = self._loadResults()
        timings = {"DPC++ (AOT)": {}, "DPC++ (JIT)": {},  "OpenMP": {}}

        for testConfig, results in results.items():
            isAot, _, problemConfig, _ = testConfig

            problemConfig = dict(problemConfig)
            graphSize = int(dict(problemConfig["kwargs"])["-n"])
            if isAot[1] is True:
                syclKey = "DPC++ (AOT)"
            else:
                syclKey = "DPC++ (JIT)"

            timings[syclKey][graphSize] = results["SYCL"]["Average Total Time (in s)"]
            timings["OpenMP"][graphSize] = results["OpenMP"]["Average Total Time (in s)"]

        ## TODO: Add standard deviation grayed out area to plot
        graphSizes = sorted(list(self.defaultGraphInputConfig["kwargs"]["-n"]))
        syclJitTimings = [sum(times) / len(times) for size, times in sorted(timings["DPC++ (JIT)"].items())]
        syclAotTimings = [sum(times) / len(times) for size, times in sorted(timings["DPC++ (AOT)"].items())]
        ompTimings = [sum(times) / len(times) for size, times in sorted(timings["OpenMP"].items())]
        
        ## Plot the absolute timings
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)

        ax.plot(graphSizes, syclJitTimings, label="DPC++ (JIT)", marker="x")
        ax.plot(graphSizes, syclAotTimings, label="DPC++ (AOT)", marker="x")
        ax.plot(graphSizes, ompTimings, label="OpenMP")

        plt.title("Absolute Performance Timings (All Resources Available)")
        plt.xlabel("Graph Vertex Count")
        plt.ylabel("Time (s)")
        ax.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=updateProp)})

        fig.savefig("absolute_timings_diagram.png", bbox_inches="tight")


        ## Plot the relative timings (Prefer this)

        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        
        relativeJitTimes = [syclJitTimings[i] / ompTimings[i] for i in range(len(syclJitTimings))]
        relativeAotTimes = [syclAotTimings[i] / ompTimings[i] for i in range(len(syclAotTimings))]
        ax.plot(graphSizes, relativeJitTimes, label="DPC++ (JIT) against OpenMP", linestyle="solid", linewidth=3.0, markersize=10, markerfacecolor="black", alpha=1, marker="x")
        ax.plot(graphSizes, relativeAotTimes, label="DPC++ (AOT) against OpenMP", linestyle="dashed",  linewidth=3.0, markersize=10, markerfacecolor="black", alpha=1, dashes=(7,7), marker="x")

        # plt.title("Relative Performance Timing (All Resources Available)")
        plt.xlabel("Graph Vertex Count")
        plt.ylabel("Slowdown (x)")
        plt.xticks(graphSizes)
                
        box = ax.get_position()
        box = ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 1])
        ax.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=updateProp)})
        
        fig.savefig("relative_timings_diagram.png", bbox_inches="tight")


## For weak scaling, we want to generate workloads

def main():
    # v = MiniviteSingleNodeWeakScaler()
    # v.run()
    # v.weakScaling()

    # v = MiniviteSingleNodeStrongScaler()
    # v.run()
    # v.strongScaling()

    v = MiniviteSingleNodeTiming()
    #v.run()
    v.timeGraphSizes()


if __name__ == "__main__":
    main()
