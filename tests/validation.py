## This program validates the correctness of the SYCL-based miniVITE program
## against the original OpenMP-based miniVITE program

## NOTE: This file requires the original miniVITE source directory to be stored
## within the same directory as this source-project (miniVITE-SYCL)
## For example: They can both be in a directory called workspace
## workspace/miniVite
## workspace/miniVITE-SYCL

from numbers import Number
from typing import List, Tuple, Dict, Optional, Iterator, Any, TypeVar
import statistics

import scipy

from _miniviteTestUtilities import MiniviteVariantTester, config

class MiniviteCorrectnessValidator(MiniviteVariantTester):
    results_location = "correctness.json"
    repeat_tests = 1

    defaultComputeConfig = {
        "MAX_MPI_RANKS": [1,],
        "MAX_NUM_THREADS": [None,]
    }

    ## NOTE: The graph.hpp logic was not modified. Therefore, we only perform the below tests
    defaultGraphInputConfig = {
        "kwargs": {
            "-n": [4000]
        },
        "args": [None,]
    }

    resultAttributesSelected = {"Modularity", "Iterations"}

    def _collectTestResults(self, graphConfig: config, computeConfig: config, repeats: int = 5) -> Dict[str, Dict[str, Number]]:
        testResults = self._performTest(graphConfig, computeConfig, self.repeat_tests)

        formattedResults = {}
        formattedResults["SYCL"] = {k:v for k, v in testResults["SYCL"].items() if k in self.resultAttributesSelected}
        formattedResults["OpenMP"] = {k:v for k, v in testResults["OpenMP"].items() if k in self.resultAttributesSelected}
        return formattedResults



    ## we find what properties of these configurations are different


    def _getGraphConfigChanges(self, graphConfig1: config, graphConfig2: config) -> List[str]:
        ## Any number of changes is still considered "a single change" because
        ## we are moving from one graph input --> to another graph input
        if graphConfig1 != graphConfig2:
            return [graphConfig1]
        return []

    def _getComputeConfigChanges(self, computeConfig1: config, computeConfig2: config) -> List[str]:
        ## NOTE We are getting the changes from config2 --> config1
        computeConfig1 = set(computeConfig1)
        computeConfig2 = set(computeConfig2)
        return list(computeConfig1.difference(computeConfig2))



    def _getCompileConfigChanges(self, compileConfig1: str, compileConfig2: str) -> List[str]:
        ## NOTE We are getting the changes from config2 --> config1
        changes = []

        ## Fix issues where a combined parameter gets split
        ## NOTE: This is bad practice to resolve this issue here!
        if "-DUSE_MPI_RMA" in compileConfig1 and "-DUSE_MPI_ACCUMULATE" in compileConfig1:
            compileConfig1 = list(compileConfig1)
            compileConfig1.remove("-DUSE_MPI_RMA")
            compileConfig1.remove("-DUSE_MPI_ACCUMULATE")
            compileConfig1.append("-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE")

        if "-DUSE_MPI_RMA" in compileConfig2 and "-DUSE_MPI_ACCUMULATE" in compileConfig2:
            compileConfig2 = list(compileConfig2)
            compileConfig2.remove("-DUSE_MPI_RMA")
            compileConfig2.remove("-DUSE_MPI_ACCUMULATE")
            compileConfig2.append("-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE")

        configDifferences = set(compileConfig1).symmetric_difference(set(compileConfig2))

        ## We now remove for MPI differences
        mpiCompileFlags = set(self.MPIFlags)
        mpiDifferences = configDifferences.intersection(mpiCompileFlags)
        assert 0 <= len(mpiDifferences) <= 2
        ## if len(mpiDifferences) == 0, then no MPI config change
        ## if len(mpiDifferences) == 2, then an MPI config must have been swapped
        ## if len(mpiDifferences) == 1, then we've gone from MPI config, to None (default) or vice versa)

        newMpiChanges = mpiDifferences.difference(compileConfig2)
        if len(newMpiChanges) == 0:
            ## if now selected default or if we kept same flag

            ## if kept the same flag, then no changes
            if mpiCompileFlags.intersection(compileConfig2) == mpiCompileFlags.intersection(compileConfig1):
                pass
            ## if we now select default, then we add an empty tuple as a change
            else:
                changes.extend(["Default MPI Comms",])
        elif len(newMpiChanges) == 1:
            changes.extend(newMpiChanges)
        else:
            raise Exception("There should be no more than 1 new MPI change!")

        ## TODO: Need to look at Misc Flags  + DS Flags
        otherCompileFlags = set(self.miscFlags).union(set(self.DSFlags))
        otherDifferences = configDifferences.intersection(otherCompileFlags)
        newOtherChanges = otherDifferences.difference(compileConfig2)
        changes.extend(otherDifferences)

        return changes

    # def _getCompileConfigChanges(self, compileConfig1: str, compileConfig2: str) -> List[str]:
    #     ## NOTE We are getting the changes from config2 --> config1
    #     changes = []

    #     # compileConfigSet1 = {s.strip() for s in compileConfig1.split(" ")}
    #     # compileConfigSet2 = {s.strip() for s in compileConfig2.split(" ")}
    #     # configDifferences = set(compileConfig1).symmetric_difference(set(compileConfig2))

    #     ## We now remove for MPI differences
    #     mpiCompileFlags = set(self.MPIFlags)
    #     mpiDifferences = configDifferences.intersection(mpiCompileFlags)
    #     assert 0 <= len(mpiDifferences) <= 2
    #     newMpiChanges = mpiDifferences.difference(compileConfigSet2)
    #     changes.extend(mpiDifferences)

    #     ## TODO: Need to look at Misc Flags  + DS Flags
    #     otherCompileFlags = set(self.miscFlags).union(set(self.DSFlags))
    #     otherDifferences = configDifferences.intersection(otherCompileFlags)
    #     newOtherChanges = otherDifferences.difference(compileConfigSet2)
    #     changes.extend(otherDifferences)

    #     return changes

        ## if there is a single addition to the MiscFlags or DSFlags (and no other changes elsewhere)
        ## - then that is a valid exclusionConfig

        ## if there is multiple additions to either MiscFlags or DSFlags
        ## - then it is not a valid exclusionConfig
        ## - we don't care about multiple additions because this makes it harder to pinpoint correctness properties

        ## if there is a single swap to miscFlags, (and no other changes)
        ## - then it is a valid exclusionConfig
        ## - we don't care about multiple additions because

        ## if there is a single swap to MPI, then that is fine (add + removal) or an add (from default)
        ## - then this would be a valid exclusion


    def generateMetrics(self) -> Dict[str, float]:
        ## For each attribute we need to find configurations that use it, against configurations that do not, but are otherwise identical
        consistencyTable = {}
        results = self._loadResults()

        keys = []
        keys.extend([x for x in self.MPIFlags if x is not None])
        keys.extend(self.DSFlags)
        keys.extend(self.miscFlags)
        
        ## NOTE: We only care about SYCL configurations and their equivalent in OpenMP
        exclusionConfigurations = {}

        ## We accumulate all the exclusionConfigs
        counter = 0
        for testcase1 in results:
            graphConfig1, compileConfig1, computeConfig1 = testcase1
            for testcase2 in results:
                counter+=1
                if testcase1 == testcase2:
                    continue

                ## We retrieve all changes in in each subconfiguration of the testcase2

                changes = []
                graphConfig2, compileConfig2, computeConfig2 = testcase2

                graphConfigChanges = self._getGraphConfigChanges(graphConfig1, graphConfig2)
                changes.extend(graphConfigChanges)
                if len(changes) > 1:
                    continue

                compileConfigChanges = self._getCompileConfigChanges(compileConfig1, compileConfig2)
                changes.extend(compileConfigChanges)
                if len(changes) > 1:
                    continue

                computeConfigChanges = self._getComputeConfigChanges(computeConfig1, computeConfig2)
                changes.extend(computeConfigChanges)
                if len(changes) > 1:
                    continue

                if len(changes) != 1:
                    ## this means there are 0 changes (which means our "if testcase1 == testcase2" condition failed)
                    raise ValueError("One of the above configChanges lists must have an element")

                ## At this point in the code we have a single change from testcase2 -> testcase1 (otherwise we have exited the for loop)
                ## We determine what the unique change is below (in case we need this later)

                newAttribute = None
                newAttributeType = None
                if len(graphConfigChanges) == 1:
                    newAttribute = graphConfigChanges[0]
                    newAttributeType = "Graph"
                elif len(computeConfigChanges) == 1:
                    newAttribute = computeConfigChanges[0]
                    newAttributeType = "Compute"
                elif len(compileConfigChanges) == 1:
                    newAttribute = compileConfigChanges[0]
                    newAttributeType = "Compile"

                ## We now save the exclusion in the data structure
                if newAttribute not in exclusionConfigurations:
                    exclusionConfigurations[newAttribute] = {}

                if testcase1 not in exclusionConfigurations[newAttribute]:
                    exclusionConfigurations[newAttribute][testcase1] = set()
                
                exclusionConfigurations[newAttribute][testcase1].add(testcase2)
                    
        print(exclusionConfigurations)
        print(counter)
        print(len(results) ** 2)
        ## We then calculate the distance for each flag
        for newAttribute, configs in exclusionConfigurations.items():
            attributeResults = {
                                    "OpenMP - Results Change": {"s.d.": [], "mean": [], "divergence": []},
                                    "DPC++ - Results Change": {"s.d.": [], "mean": [], "divergence": []},
                                    "OpenMP and DPC++ - similarity of result change": {"s.d.": [], "mean": [], "divergence": []},
                                    "OpenMP and DPC++ - similarity of final result": {"s.d.": [], "mean": [], "divergence": []},
                                }

            for inclusion, exclusions in configs.items():
                ## Each test config can have many configs that a csingle change can be applied to create it
                ## We'll need to average our metrics.

                for exclusion in exclusions:
                    ## We want to calculate similarity between the newly added attribute

                    ## The below four variables contain a list of modularities on
                    ## configuration Runs for OpenMP and SYCL
                    inclusionSYCLMods = results[inclusion]["SYCL"]["Modularity"]
                    inclusionOMPMods = results[inclusion]["OpenMP"]["Modularity"]
                    exclusionSYCLMods = results[exclusion]["SYCL"]["Modularity"]
                    exclusionOMPMods = results[exclusion]["OpenMP"]["Modularity"]

                    inclusionOMPMean = statistics.mean(inclusionOMPMods)
                    inclusionOMPStdev = statistics.stdev(inclusionOMPMods)
                    exclusionOMPMean = statistics.mean(exclusionOMPMods)
                    exclusionOMPStdev = statistics.stdev(exclusionOMPMods)
                    OMPStdevDifference = inclusionOMPstdev - exclusionOMPStdev
                    OMPMeanDifference = inclusionOMPMean - exclusionOMPMean
                    OMPwassersteinDistance = scipy.wasserstein_distance(inclusionOMPMods, exclusionOMPMods)

                    attributeResults["OpenMP - Results Change"]["s.d."].append(OMPStdevDifference)
                    attributeResults["OpenMP - Results Change"]["mean"].append(OMPMeanDifference)
                    attributeResults["OpenMP - Results Change"]["divergence"].append(OMPwassersteinDistance)

                    inclusionSYCLMean = statistics.mean(inclusionSYCLMods)
                    inclusionSYCLStdev = statistics.stdev(inclusionSYCLMods)
                    exclusionSYCLMean = statistics.mean(exclusionSYCLMods)
                    exclusionSYCLStdev = statistics.stdev(exclusionSYCLMods)
                    SYCLStdevDifference = inclusionSYCLstdev - exclusionSYCLStdev
                    SYCLMeanDifference = inclusionSYCLMean - exclusionSYCLMean
                    SYCLwassersteinDistance = scipy.wasserstein_distance(inclusionSYCLMods, exclusionSYCLMods)

                    attributeResults["DPC++ - Results Change"]["s.d."].append(SYCLStdevDifference)
                    attributeResults["DPC++ - Results Change"]["mean"].append(SYCLMeanDifference)
                    attributeResults["DPC++ - Results Change"]["divergence"].append(SYCLwassersteinDistance)

                    attributeResults["OpenMP and DPC++ - similarity of result change"]["s.d."].append(SYCLStdevDifference - OMPStdevDifference)
                    attributeResults["OpenMP and DPC++ - similarity of result change"]["mean"].append(SYCLMeanDifference - OMPMeanDifference)
                    attributeResults["OpenMP and DPC++ - similarity of result change"]["divergence"].append(SYCLwassersteinDistance - OMPwassersteinDistance)

                    attributeResults["OpenMP and DPC++ - similarity of final result"]["s.d."].append(inclusionOMPStdev - inclusionOMPStdev)
                    attributeResults["OpenMP and DPC++ - similarity of final result"]["mean"].append(exclusionSYCLMean - inclusionOMPMean)
                    attributeResults["OpenMP and DPC++ - similarity of final result"]["divergence"].append(scipy.wasserstein_distance(inclusionSYCLMods, inclusionOMPMods))

                ## End For Loop

                for comparison, results in attributeResults.items():
                    for measures in results:
                        results[measures] = statistics.mean(results[measures])



                    ## Each config (SYCL or OpenMP), will have a list of modularity values
                    
                    ## Let the following:
                    ## - config1 is the original config
                    ## - config2 is a config such that one changed applied to config1 allows you to get config2

                    ## We want to measure
                    ## - how consistent are the modularity results of (Model X config 1 --> Model X config2) for both OpenMP and SYCL seperately? 
                    ##      - are the new results more / less consistent (i.e. s.d.?)
                    ##          - s.d.?
                    ##          - mean?
                    ##          - skewness / distribution shape?
                    ## - how similar is the impact of (Model X config 1 --> Model X config 2) and (Model Y config 1 --> Model Y config 2)
                    ##      - are the impacts to the original distributions similar?
                    ##          - s.d.?
                    ##          - mean?
                    ##          - skewness / distribution shape?
                    ## - how similar are the results of (Model X config 2) and (Model Y config 2)
                    ##      - are the distributions similar?
                    ##          - s.d.?
                    ##          - mean?
                    ##          - skewness / distribution shape?

                    ## TODO: Are there any measure I'm missing here?
                    ## TODO: How would I "average" these results, would it be ok to just average it ordinally?v  

                    ## NOTE: Modularity has a range (-1/2, 1), where a modularity > 0 means bebuctter than random chance
                    ## - SOme sources say the range is (-1, 1)

                    # scipy.stats.wasserstein_distance(x, y)


            ## here we calculate some val
            consistencyTable[newAttribute] = attributeResults
        

        print(consistencyTable)

        ## For each flag and its corresponding exclusionConfigs,
        ## we "compare" their results in OpenMP and SYCL

    



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





def main():
    v = MiniviteCorrectnessValidator()
    v.run()
    v.generateMetrics()



if __name__ == "__main__":
    main()