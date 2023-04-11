// ***********************************************************************
//
//                              miniVite
//
// ***********************************************************************
//
//       Copyright (2018) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************ 

#pragma once
#ifndef DSPL_HPP
#define DSPL_HPP

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <mpi.h>
#include <CL/sycl.hpp>
#include "graph.hpp"
#include "utils.hpp"

struct Comm {
  GraphElem size;
  GraphWeight degree;

  Comm() : size(0), degree(0.0) {};
  Comm(GraphElem size, GraphWeight degree): size(size), degree(degree) {};
};

struct CommInfo {
    GraphElem community;
    GraphElem size;
    GraphWeight degree;
};

const int SizeTag           = 1;
const int VertexTag         = 2;
const int CommunityTag      = 3;
const int CommunitySizeTag  = 4;
const int CommunityDataTag  = 5;

static MPI_Datatype commType;

// We define USM STL allocators
typedef sycl::usm_allocator<GraphWeight, sycl::usm::alloc::shared> vec_gw_alloc;
typedef sycl::usm_allocator<GraphElem, sycl::usm::alloc::shared> vec_ge_alloc;
typedef sycl::usm_allocator<bool, sycl::usm::alloc::shared> vec_bool_alloc;
typedef sycl::usm_allocator<Comm, sycl::usm::alloc::shared> vec_comm_alloc;
typedef sycl::usm_allocator<CommInfo, sycl::usm::alloc::shared> vec_commi_alloc;
typedef sycl::usm_allocator<int, sycl::usm::alloc::shared> vec_int_alloc;
typedef sycl::usm_allocator<std::vector<CommInfo>, sycl::usm::alloc::shared> vec_vec_commi_alloc;
typedef sycl::usm_allocator<std::vector<GraphElem>, sycl::usm::alloc::shared> vec_vec_ge_alloc;
typedef sycl::usm_allocator<std::unordered_set<GraphElem>, sycl::usm::alloc::shared> vec_uset_ge_alloc;
// typedef sycl::usm_allocator<std::unordered_set<GraphElem, vec_ge_alloc>, sycl::usm::alloc::shared> vec_uset_ge_alloc;

// Defined a SYCL queue using the CPU selector
#ifdef GPU_DEVICE
sycl::queue q{sycl::gpu_selector_v};
#else
sycl::queue q{sycl::cpu_selector_v};
#endif

// Variable for setting
int threadCount;
int maxWorkGroupSize;
int minWorkGroupSize;
int maxReductionWorkGroupSize;

#define getWorkGroupSize(workItemCount) \
std::max(std::min(((int) std::ceil( (double) workItemCount / (double) threadCount)), maxWorkGroupSize), minWorkGroupSize)
// size = min(max(ceil(len(workItems) / threadCount, minWorkGroupSize), maxWorkGroupSize)

#define getReductionWorkGroupSize(workItemCount) std::min(getWorkGroupSize(workItemCount), maxReductionWorkGroupSize)

// we instantiate USM STL allocators (dependency on sycl::queue q)
vec_gw_alloc vec_gw_allocator(q);
vec_ge_alloc vec_ge_allocator(q);
vec_comm_alloc vec_comm_allocator(q);
vec_commi_alloc vec_commi_allocator(q);
vec_vec_commi_alloc vec_vec_commi_allocator(q);
vec_int_alloc vec_int_allocator(q);
vec_vec_ge_alloc vec_vec_ge_allocator(q);
vec_uset_ge_alloc vec_uset_ge_allocator(q);

void distSumVertexDegree(const Graph &g, std::vector<GraphWeight, vec_gw_alloc> &vDegree, std::vector<Comm, vec_comm_alloc> &localCinfo, sycl::queue &q)
{

  // we then create pointers to the underlying data
  auto _vDegree = vDegree.data();
  auto _localCinfo = localCinfo.data();
  const Graph *_g = &g;

  const GraphElem nv = g.get_lnv();

  q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
    const int workGroupSize = getWorkGroupSize(nv);
    h.parallel_for(sycl::nd_range<1>{nv, workGroupSize}, [=](sycl::nd_item<1> item){
      int i = item.get_global_id();
#else
    h.parallel_for(nv, [=](sycl::id<1> i){
#endif
      GraphElem e0, e1;
      GraphWeight tw = 0.0;

      _g->edge_range(i, e0, e1);
      for (GraphElem k = e0; k < e1; k++) {
        const Edge &edge = _g->get_edge(k);
        tw += edge.weight_;
      }

      _vDegree[i] = tw;
    
      _localCinfo[i].degree = tw;
      _localCinfo[i].size = 1L;
    });
  }).wait();

}


GraphWeight distCalcConstantForSecondTerm(const std::vector<GraphWeight, vec_gw_alloc> &vDegree, MPI_Comm gcomm, sycl::queue &q)
{

  GraphWeight totalEdgeWeightTwice = 0.0;
  int me = -1;

  const size_t vsz = vDegree.size();
  const int workGroupSize = getReductionWorkGroupSize(vsz);
  // const int workGroupSize = 4;

  auto _vDegree = vDegree.data();
  GraphWeight localWeight = 0;
  GraphWeight *usm_localWeight = sycl::malloc_shared<GraphWeight>(1, q);
  *usm_localWeight = 0;

  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{vsz, workGroupSize},
      sycl::reduction(usm_localWeight, std::plus<>()),
      [=](sycl::nd_item<1> it, auto& usm_localWeight) {
        int i = it.get_global_id(0);
        usm_localWeight += _vDegree[i];
    });
  }).wait();


  localWeight = *usm_localWeight;
  // We free the USM memory
  sycl::free(usm_localWeight, q);

  // Global reduction
  MPI_Allreduce(&localWeight, &totalEdgeWeightTwice, 1, 
          MPI_WEIGHT_TYPE, MPI_SUM, gcomm);

  // ... and finally return this constant
  return (1.0 / static_cast<GraphWeight>(totalEdgeWeightTwice));
} // distCalcConstantForSecondTerm



void distInitComm(std::vector<GraphElem, vec_ge_alloc> &pastComm, 
                  std::vector<GraphElem, vec_ge_alloc> &currComm, 
                  const GraphElem base, sycl::queue &q)
{
  const size_t csz = currComm.size();

#ifdef DEBUG_ASSERTIONS  
  assert(csz == pastComm.size());
#endif

  auto _pastComm = pastComm.data();
  auto _currComm = currComm.data();

  q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
    const int workGroupSize = getWorkGroupSize(csz);
    h.parallel_for(sycl::nd_range<1>{csz, workGroupSize}, [=](sycl::nd_item<1> item){
      int i = item.get_global_id();
#else
    h.parallel_for(csz, [=](sycl::id<1> i){
#endif
      _pastComm[i] = i + base;
      _currComm[i] = i + base;
    });
  }).wait();

} // distInitComm

void distInitLouvain(const Graph &dg, std::vector<GraphElem, vec_ge_alloc> &pastComm, 
        std::vector<GraphElem, vec_ge_alloc> &currComm, std::vector<GraphWeight, vec_gw_alloc> &vDegree, 
        std::vector<GraphWeight, vec_gw_alloc> &clusterWeight, std::vector<Comm, vec_comm_alloc> &localCinfo, 
        std::vector<Comm, vec_comm_alloc> &localCupdate, GraphWeight &constantForSecondTerm,
        const int me)
{
  const GraphElem base = dg.get_base(me);
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  vDegree.resize(nv);
  pastComm.resize(nv);
  currComm.resize(nv);
  clusterWeight.resize(nv);
  localCinfo.resize(nv);
  localCupdate.resize(nv);

  distSumVertexDegree(dg, vDegree, localCinfo, q);
  constantForSecondTerm = distCalcConstantForSecondTerm(vDegree, gcomm, q);
  distInitComm(pastComm, currComm, base, q);
  
} // distInitLouvain

GraphElem distGetMaxIndex(const std::vector<GraphElem> &clmap, 
                              const std::vector<GraphWeight> &counter,
			                        const GraphWeight selfLoop, 
                              const Comm localCinfo[], 
			                        const Comm remoteCinfo[], 
                              const GraphWeight vDegree, 
                              const GraphElem currSize, 
                              const GraphWeight currDegree, 
                              const GraphElem currComm,
			                        const GraphElem base, 
                              const GraphElem bound, 
                              const GraphWeight constant)
{
  GraphElem maxIndex = currComm;
  GraphWeight curGain = 0.0, maxGain = 0.0;
  GraphWeight eix = static_cast<GraphWeight>(counter[0]) - static_cast<GraphWeight>(selfLoop);

  GraphWeight ax = currDegree - vDegree;
  GraphWeight eiy = 0.0, ay = 0.0;

  GraphElem maxSize = currSize; 
  GraphElem size = 0;

  GraphElem vertexIndex = 0;
  auto iter = clmap.begin();
#ifdef DEBUG_ASSERTIONS  
  assert(iter != clmap.end());
#endif
  do {
      GraphElem storedAlready = *iter;
      // if storedAlready != -1, then we proceed (this means it's empty and we can skip)
      if (storedAlready >= 0 && currComm != vertexIndex) {

          // is_local, direct access local info
          if ((vertexIndex >= base) && (vertexIndex < bound)) {
              ay = localCinfo[vertexIndex-base].degree;
              size = localCinfo[vertexIndex - base].size;   
          }
          else {
              // is_remote, lookup map
              Comm remote_comm = remoteCinfo[vertexIndex];
              ay = remote_comm.degree;
              size = remote_comm.size; 
          }

          eiy = counter[storedAlready];

          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant;

          if ((curGain > maxGain) || ((curGain == maxGain) && (curGain != 0.0) && (vertexIndex < maxIndex))) {
              maxGain = curGain;
              maxIndex = vertexIndex;
              maxSize = size;
          }
      }
      iter++;
      vertexIndex++;
  } while (iter != clmap.end());

  if ((maxSize == 1) && (currSize == 1) && (maxIndex > currComm))
    maxIndex = currComm;

  return maxIndex;
} // distGetMaxIndex

#ifdef DEBUG_ASSERTIONS
GraphWeight distBuildLocalMapCounter(const GraphElem e0, const GraphElem e1, 
                                    std::vector<GraphElem> &clmap, 
				                            std::vector<GraphWeight> &counter, 
                                    GraphElem &counter_size,
                                    const Graph *g, 
                                    const GraphElem* currComm,
                                    int currCommSize,
                                    const GraphElem* remoteComm,
                                    int remoteCommSize,
	                                  const GraphElem vertex, const GraphElem base, const GraphElem bound)
#else
GraphWeight distBuildLocalMapCounter(const GraphElem e0, const GraphElem e1, 
                                    std::vector<GraphElem> &clmap, 
				                            std::vector<GraphWeight> &counter, 
                                    GraphElem &counter_size,
                                    const Graph *g, 
                                    const GraphElem* currComm,
                                    const GraphElem* remoteComm,
	                                  const GraphElem vertex, const GraphElem base, const GraphElem bound)
#endif
{
  GraphElem numUniqueClusters = 1L;
  GraphWeight selfLoop = 0;

  for (GraphElem j = e0; j < e1; j++) {
    const Edge &edge = g->get_edge(j);
    const GraphElem &tail_ = edge.tail_;
    const GraphWeight &weight = edge.weight_;

    if (tail_ == vertex + base)
      selfLoop += weight;

    // is_local, direct access local std::vector<GraphElem>
    GraphElem tcomm;
    if ((tail_ >= base) && (tail_ < bound)){
#ifdef DEBUG_ASSERTIONS
      assert(0 <= (tail_ - base) && (tail_ - base) < currCommSize);
#endif
      tcomm = currComm[tail_ - base];
    }
    else { // is_remote, lookup map
#ifdef DEBUG_ASSERTIONS
      assert(0 <= tail_ && tail_ < remoteCommSize);
#endif
      tcomm = remoteComm[tail_];
#ifdef DEBUG_ASSERTIONS
      assert(tcomm != -1); // -1 means not there in the vector - (remoteComm has been replaced with a vector from a unordered_map)
#endif
    }

#ifdef DEBUG_ASSERTIONS
    assert (0 <= tcomm && tcomm < clmap.size());
#endif
    const GraphElem storedAlready = clmap[tcomm];
    
    if (storedAlready != -1){
#ifdef DEBUG_ASSERTIONS
      assert (0 <= storedAlready && storedAlready < counter.size());
#endif
      counter[storedAlready] += weight;
    }
    else {
#ifdef DEBUG_ASSERTIONS
        assert (0 <= tcomm && tcomm < clmap.size());
        assert (0 <= counter_size && counter_size < counter.size());
#endif
        clmap[tcomm] = numUniqueClusters;
        counter[counter_size] = weight;
        counter_size++;
        numUniqueClusters++;
    }
  }

  return selfLoop;
} // distBuildLocalMapCounter


void distExecuteLouvainIteration(const GraphElem nv, const Graph &dg, const std::vector<GraphElem, vec_ge_alloc> &currComm,
				                              std::vector<GraphElem, vec_ge_alloc> &targetComm, const std::vector<GraphWeight, vec_gw_alloc> &vDegree,
                                      std::vector<Comm, vec_comm_alloc> &localCinfo, std::vector<Comm, vec_comm_alloc> &localCupdate,
				                              const std::vector<GraphElem, vec_ge_alloc> &remoteComm, 
                                      const std::vector<Comm, vec_comm_alloc> &remoteCinfo, 
                                      std::vector<Comm, vec_comm_alloc> &remoteCupdate, const GraphWeight constantForSecondTerm,
                                      std::vector<GraphWeight, vec_gw_alloc> &clusterWeight, const int me){


  // Access to underlying memory blocks for vectors
  auto _currComm = currComm.data();
  auto _targetComm = targetComm.data();
  auto _vDegree = vDegree.data();
  auto _localCinfo = localCinfo.data();
  auto _clusterWeight = clusterWeight.data();
  auto _remoteComm = remoteComm.data();
  auto _remoteCinfo = remoteCinfo.data();
  auto _remoteCupdate = remoteCupdate.data();
  auto _localCupdate = localCupdate.data();

#ifdef DEBUG_ASSERTIONS
  int _localCupdateSize = localCupdate.size();
  int _vDegreeSize = vDegree.size();
  int _clusterWeightSize = clusterWeight.size();
  int _currCommSize = currComm.size();
  int _remoteCommSize = remoteComm.size();
  int _targetCommSize = targetComm.size();
  int _remoteCupdateSize = remoteCupdate.size();
  int _localCinfoSize = localCinfo.size();
  int _remoteCinfoSize = remoteCinfo.size();
#endif

  const Graph *_dg = &dg;

  q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
    const int workGroupSize = getWorkGroupSize(nv);
    h.parallel_for(sycl::nd_range<1>{nv, workGroupSize}, [=](sycl::nd_item<1> item){
      int i = item.get_global_id();
#else
    h.parallel_for(nv, [=](sycl::id<1> i){
#endif
      GraphElem localTarget = -1;
      GraphElem e0, e1, selfLoop = 0;
      
      int max_neighbors = _dg->get_nv();

      std::vector<GraphWeight> counter(max_neighbors, 0.0);
      GraphElem counter_size = 0;

      std::vector<GraphElem> clmap(max_neighbors, -1); 

      const GraphElem base = _dg->get_base(me), bound = _dg->get_bound(me);
      const GraphElem cc = _currComm[i];
      GraphWeight ccDegree;
      GraphElem ccSize;  
      bool currCommIsLocal = false; 
      bool targetCommIsLocal = false;

      // Current Community is local
      if (cc >= base && cc < bound) {
#ifdef DEBUG_ASSERTIONS
        assert (0 <= (cc - base) && (cc - base) < _localCinfoSize);
#endif
        ccDegree=_localCinfo[cc-base].degree;
        ccSize=_localCinfo[cc-base].size;
        currCommIsLocal=true;
      } else {
        // is remote
#ifdef DEBUG_ASSERTIONS
        assert (0 <= (cc) && (cc) < _remoteCinfoSize);
#endif
        Comm comm = _remoteCinfo[cc];
        ccDegree = comm.degree;
        ccSize = comm.size;
        currCommIsLocal=false;
      }

      _dg->edge_range(i, e0, e1);

#ifdef DEBUG_ASSERTIONS
      assert (0 <= i && i < _vDegreeSize);
#endif

      if (e0 != e1) {
#ifdef DEBUG_ASSERTIONS
        assert(0 <= cc && cc < clmap.size());
        assert(0 <= i && i < _clusterWeightSize);
#endif
        clmap[cc] = 0;
        counter_size++;

        // modified counter, counter_size, clmap
#ifdef DEBUG_ASSERTIONS
        selfLoop = distBuildLocalMapCounter(e0, e1, clmap, counter, counter_size, _dg, 
                                                  _currComm, _currCommSize, _remoteComm, _remoteCommSize, i, base, bound);
#else 
        selfLoop = distBuildLocalMapCounter(e0, e1, clmap, counter, counter_size, _dg, 
                                                  _currComm, _remoteComm, i, base, bound);
#endif

        _clusterWeight[i] += counter[0];

        // no modifications
        localTarget = distGetMaxIndex(clmap, counter, selfLoop, _localCinfo, _remoteCinfo, 
                        _vDegree[i], ccSize, ccDegree, cc, base, bound, constantForSecondTerm);
      }
      else
        localTarget = cc;

#ifdef DEBUG_ASSERTIONS
      assert(0 <= localTarget);
      assert(0 <= (cc - base) && (cc - base) < _localCupdateSize); 	
      assert(0 <= (localTarget - base) && (localTarget - base) < _localCupdateSize); 
#endif

      // create atomic references (replaces #omp pragma atomic update)
      sycl::atomic_ref<GraphWeight, sycl::memory_order::relaxed, sycl::memory_scope::system> localTarget_base_degree(_localCupdate[localTarget-base].degree);
      sycl::atomic_ref<GraphElem, sycl::memory_order::relaxed, sycl::memory_scope::system> localTarget_base_size(_localCupdate[localTarget-base].size);
      sycl::atomic_ref<GraphWeight, sycl::memory_order::relaxed, sycl::memory_scope::system> cc_base_degree(_localCupdate[cc-base].degree);
      sycl::atomic_ref<GraphElem, sycl::memory_order::relaxed, sycl::memory_scope::system> cc_base_size(_localCupdate[cc-base].size);
      
      // is the Target Local?
      if (localTarget >= base && localTarget < bound)
        targetCommIsLocal = true;

      // current and target comm are local - atomic updates to vectors
      if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && targetCommIsLocal) {
#ifdef DEBUG_ASSERTIONS
        assert(base <= localTarget && localTarget < bound);
        assert(base <= cc && cc < bound);
#endif
        localTarget_base_degree += _vDegree[i];
        localTarget_base_size++;
        cc_base_degree -= _vDegree[i];
        cc_base_size--;
      }	

      // current is local, target is not - do atomic on local, accumulate in Maps for remote
      if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && !targetCommIsLocal) {
#ifdef DEBUG_ASSERTIONS
        assert(0 <= localTarget && localTarget < _remoteCupdateSize);
        assert(0 <= i && i < _vDegreeSize);
#endif
        cc_base_degree -= _vDegree[i];
        cc_base_size--;

        // search target!
        Comm target_comm = _remoteCupdate[localTarget];
        sycl::atomic_ref<GraphElem, sycl::memory_order::relaxed, sycl::memory_scope::system> target_comm_size(target_comm.size);
        sycl::atomic_ref<GraphWeight, sycl::memory_order::relaxed, sycl::memory_scope::system> target_comm_degree(target_comm.degree);
        
        target_comm_degree += _vDegree[i];
        target_comm_size++;
      }
            
      // current is remote, target is local - accumulate for current, atomic on local
      if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && targetCommIsLocal) {
#ifdef DEBUG_ASSERTIONS
        assert(0 <= localTarget && localTarget < _remoteCupdateSize);
        assert(0 <= i && i < _vDegreeSize);
#endif
        localTarget_base_degree += _vDegree[i];
        localTarget_base_size++;
      
        // search current 
        Comm current_comm = _remoteCupdate[cc];
        sycl::atomic_ref<GraphElem, sycl::memory_order::relaxed, sycl::memory_scope::system> current_comm_size(current_comm.size);
        sycl::atomic_ref<GraphWeight, sycl::memory_order::relaxed, sycl::memory_scope::system> current_comm_degree(current_comm.degree);
        
        current_comm_degree -= _vDegree[i];
        current_comm_size--;
      }
                        
      // current and target are remote - accumulate for both
      if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && !targetCommIsLocal) {
#ifdef DEBUG_ASSERTIONS
        assert(0 <= localTarget && localTarget < _remoteCupdateSize);
        assert(0 <= i && i < _vDegreeSize);
#endif
        // search current 
        Comm current_comm = _remoteCupdate[cc];
        sycl::atomic_ref<GraphElem, sycl::memory_order::relaxed, sycl::memory_scope::system> current_comm_size(current_comm.size);
        sycl::atomic_ref<GraphWeight, sycl::memory_order::relaxed, sycl::memory_scope::system> current_comm_degree(current_comm.degree);

        current_comm_degree -= _vDegree[i];
        current_comm_size--;
  
        // search target
        Comm target_comm = _remoteCupdate[localTarget];
        sycl::atomic_ref<GraphElem, sycl::memory_order::relaxed, sycl::memory_scope::system> target_comm_size(target_comm.size);
        sycl::atomic_ref<GraphWeight, sycl::memory_order::relaxed, sycl::memory_scope::system> target_comm_degree(target_comm.degree);
        
        target_comm_degree += _vDegree[i];
        target_comm_size++;
      }

#ifdef DEBUG_ASSERTIONS
      assert(0 <= i && i < _targetCommSize);
#endif
      _targetComm[i] = localTarget;

    });
  }).wait(); 

}


GraphWeight distComputeModularity(const Graph &g, std::vector<Comm, vec_comm_alloc> &localCinfo,
			          const std::vector<GraphWeight, vec_gw_alloc> &clusterWeight,
			     const GraphWeight constantForSecondTerm,
			     const int me)
{
  const GraphElem nv = g.get_lnv();
  MPI_Comm gcomm = g.get_comm();

  GraphWeight le_la_xx[2];
  GraphWeight e_a_xx[2] = {0.0, 0.0};
  GraphWeight le_xx = 0.0, la2_x = 0.0;

#ifdef DEBUG_ASSERTIONS  
  assert((clusterWeight.size() == nv));
#endif

  auto _localCinfo = localCinfo.data();
  auto _clusterWeight = clusterWeight.data();

  GraphWeight *_le_xx = sycl::malloc_shared<GraphWeight>(1, q);
  GraphWeight *_la2_x = sycl::malloc_shared<GraphWeight>(1, q);
  *_le_xx = le_xx;
  *_la2_x = la2_x;

  // NOTE: The order of the arguments matters for the parallel_for lambda
  // This order corresponds to the order of the reductions

  // BUG: THere are some weird memory runtime errors we experience with multi-reductions
  // A workgroup size that would work on two single reductions will consistently raise
  // runtime errors if the graph input size of the program is large enough. Reducing
  // the size of the work-group will always* resolve this issue. However if you then
  // increase the size of the graph (i.e. vertices), this runtime error is observed again
  
  // *always - in our limited testing of this issue, we haven't seen any behavior that would
  // contradict the behavior we observed.

#ifdef ENABLE_SYCL_MULTI_REDUCTION
  const int workGroupSize = 4;
  q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::nd_range<1>{nv, local_group_size},
                   sycl::reduction(_le_xx, std::plus<>()),
                   sycl::reduction(_la2_x, std::plus<>()),
                   [=](sycl::nd_item<1> it, auto &_le_xx, auto &_la2_x){
                      int i = it.get_global_id(0);
                      _le_xx += _clusterWeight[i];
                      _la2_x += static_cast<GraphWeight>(_localCinfo[i].degree) * static_cast<GraphWeight>(_localCinfo[i].degree); 
    });
  }).wait();

#else
  const int workGroupSize = std::max(std::min(getWorkGroupSize(nv), maxReductionWorkGroupSize / 2), 4);
  q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::nd_range<1>{nv, workGroupSize},
                   sycl::reduction(_le_xx, std::plus<>()),
                   [=](sycl::nd_item<1> it, auto &_le_xx){
                      int i = it.get_global_id(0);
                      _le_xx += _clusterWeight[i];
    });
  });

  q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::nd_range<1>{nv, workGroupSize},
                   sycl::reduction(_la2_x, std::plus<>()),
                   [=](sycl::nd_item<1> it, auto &_la2_x){
                      int i = it.get_global_id(0);
                      _la2_x += static_cast<GraphWeight>(_localCinfo[i].degree) * static_cast<GraphWeight>(_localCinfo[i].degree); 
    });
  });
  
  q.wait();
#endif

  le_xx = *_le_xx;
  la2_x = *_la2_x;

  le_la_xx[0] = le_xx;
  le_la_xx[1] = la2_x;

  sycl::free(_le_xx, q);
  sycl::free(_la2_x, q);

#ifdef DEBUG_ASSERTIONS  
  const double t0 = MPI_Wtime();
#endif

  MPI_Allreduce(le_la_xx, e_a_xx, 2, MPI_WEIGHT_TYPE, MPI_SUM, gcomm);

#ifdef DEBUG_ASSERTIONS  
  const double t1 = MPI_Wtime();
#endif

  GraphWeight currMod = std::fabs((e_a_xx[0] * constantForSecondTerm) - 
      (e_a_xx[1] * constantForSecondTerm * constantForSecondTerm));
#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]le_xx: " << le_xx << ", la2_x: " << la2_x << std::endl;
  std::cout << "[" << me << "]e_xx: " << e_a_xx[0] << ", a2_x: " << e_a_xx[1] << ", currMod: " << currMod << std::endl;
  std::cout << "[" << me << "]Reduction time: " << (t1 - t0) << std::endl;
#endif

  return currMod;
} // distComputeModularity

void distUpdateLocalCinfo(std::vector<Comm, vec_comm_alloc> &localCinfo, const std::vector<Comm, vec_comm_alloc> &localCupdate)
{

  auto _localCinfo = localCinfo.data();
  auto _localCupdate = localCupdate.data();
  GraphElem csz = localCinfo.size();

  q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
    const int workGroupSize = getWorkGroupSize(csz);
    h.parallel_for(sycl::nd_range<1>{csz, workGroupSize}, [=](sycl::nd_item<1> item){
      int i = item.get_global_id();
#else
    h.parallel_for(csz, [=](sycl::id<1> i){
#endif
      _localCinfo[i].size += _localCupdate[i].size;
      _localCinfo[i].degree += _localCupdate[i].degree;
    });
  }).wait();
}

void distCleanCWandCU(const GraphElem nv, std::vector<GraphWeight, vec_gw_alloc> &clusterWeight, std::vector<Comm, vec_comm_alloc> &localCupdate)
{

  auto _clusterWeight = clusterWeight.data();
  auto _localCupdate = localCupdate.data();

  q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
    const int workGroupSize = getWorkGroupSize(nv);
    h.parallel_for(sycl::nd_range<1>{nv, workGroupSize}, [=](sycl::nd_item<1> item){
      int i = item.get_global_id();
#else
    h.parallel_for(nv, [=](sycl::id<1> i){
#endif
      _clusterWeight[i] = 0;
      _localCupdate[i].degree = 0;
      _localCupdate[i].size = 0;
    });
  }).wait();

} // distCleanCWandCU

#if defined(USE_MPI_RMA)
void fillRemoteCommunities(const Graph &dg, const int me, const int nprocs,
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem, vec_ge_alloc> &rsizes, const std::vector<GraphElem, vec_ge_alloc> &svdata, 
        const std::vector<GraphElem, vec_ge_alloc> &rvdata, const std::vector<GraphElem, vec_ge_alloc> &currComm, 
        const std::vector<Comm, vec_comm_alloc> &localCinfo, std::vector<Comm, vec_comm_alloc> &remoteCinfo, 
        std::vector<GraphElem, vec_ge_alloc> &remoteComm, std::vector<Comm, vec_comm_alloc> &remoteCupdate, 
        const MPI_Win &commwin, const std::vector<GraphElem> &disp)
#else
void fillRemoteCommunities(const Graph &dg, const int me, const int nprocs,
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem, vec_ge_alloc> &rsizes, const std::vector<GraphElem, vec_ge_alloc> &svdata, 
        const std::vector<GraphElem, vec_ge_alloc> &rvdata, const std::vector<GraphElem, vec_ge_alloc> &currComm, 
        const std::vector<Comm, vec_comm_alloc> &localCinfo, std::vector<Comm, vec_comm_alloc> &remoteCinfo, 
        std::vector<GraphElem, vec_ge_alloc> &remoteComm, std::vector<Comm, vec_comm_alloc> &remoteCupdate)
#endif
{
#if defined(USE_MPI_RMA)
    std::vector<GraphElem, vec_ge_alloc> scdata(ssz, vec_ge_allocator);
#else
    std::vector<GraphElem> rcdata(rsz);
    std::vector<GraphElem, vec_ge_alloc> scdata(ssz, vec_ge_allocator);
#endif
  GraphElem spos, rpos;

#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector< std::vector< GraphElem>, vec_vec_ge_alloc > rcinfo(nprocs, vec_vec_ge_allocator);
#else
  std::vector<std::unordered_set<GraphElem>, vec_uset_ge_alloc > rcinfo(nprocs, vec_uset_ge_allocator);
#endif


#if defined(USE_MPI_SENDRECV)
#else
  std::vector<MPI_Request> rreqs(nprocs), sreqs(nprocs);
#endif

#ifdef DEBUG_PRINTF  
  double t0, t1, ta = 0.0;
#endif

#if defined(USE_MPI_RMA) && !defined(USE_MPI_ACCUMULATE)
  int num_comm_procs;
#endif

#if defined(USE_MPI_RMA) && !defined(USE_MPI_ACCUMULATE)
  spos = 0;
  rpos = 0;
  std::vector<int> comm_proc(nprocs);
  std::vector<int> comm_proc_buf_disp(nprocs);
  
  /* Initialize all to -1 (unsure if necessary) */
  for (int i = 0; i < nprocs; i++) {
      comm_proc[i] = -1;
      comm_proc_buf_disp[i] = -1;
  }
  
  num_comm_procs = 0;
  for (int i = 0; i < nprocs; i++) {
      if ((i != me) && (ssizes[i] > 0)) {
          comm_proc[num_comm_procs] = i;
          comm_proc_buf_disp[num_comm_procs] = spos;
          num_comm_procs++;
      }
      spos += ssizes[i];
      rpos += rsizes[i];
  }
#endif

  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  // SYCL port
  // Collects Communities of local vertices for remote nodes
  auto _svdata = svdata.data();
  auto _scdata = scdata.data();
  auto _currComm = currComm.data();

  q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
    const int workGroupSize = getWorkGroupSize(ssz);
    h.parallel_for(sycl::nd_range<1>{ssz, workGroupSize}, [=](sycl::nd_item<1> item){
      int i = item.get_global_id();
#else
    h.parallel_for(ssz, [=](sycl::id<1> i){
#endif
      const GraphElem vertex = _svdata[i];
#ifdef DEBUG_ASSERTIONS
      assert((vertex >= base) && (vertex < bound));
#endif
      const GraphElem comm = _currComm[vertex - base];
      _scdata[i] = comm;
    });

  }).wait();

  std::vector<GraphElem> rcsizes(nprocs);
  std::vector<GraphElem, vec_ge_alloc> scsizes(nprocs, vec_ge_allocator);
  std::vector<CommInfo, vec_commi_alloc> sinfo(vec_commi_allocator);
  std::vector<CommInfo> rinfo;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
#if !defined(USE_MPI_RMA) || defined(USE_MPI_ACCUMULATE)
  spos = 0;
  rpos = 0;
#endif
#if defined(USE_MPI_COLLECTIVES)
  std::vector<int> scnts(nprocs), rcnts(nprocs), sdispls(nprocs);
  std::vector<int, vec_int_alloc> rdispls(nprocs, vec_int_allocator);

  for (int i = 0; i < nprocs; i++) {
      scnts[i] = ssizes[i];
      rcnts[i] = rsizes[i];
      sdispls[i] = spos;
      rdispls[i] = rpos;
      spos += scnts[i];
      rpos += rcnts[i];
  }
  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(scdata.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rcdata.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, gcomm);
#elif defined(USE_MPI_RMA)
#if defined(USE_MPI_ACCUMULATE)
  for (int i = 0; i < nprocs; i++) {
      if ((i != me) && (ssizes[i] > 0)) {
          MPI_Accumulate(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, 
                  disp[i], ssizes[i], MPI_GRAPH_TYPE, MPI_REPLACE, commwin);
      }
      spos += ssizes[i];
      rpos += rsizes[i];
  }
#else
  for (int i = 0; i < num_comm_procs; i++) {
      int target_rank = comm_proc[i];
      MPI_Put(scdata.data() + comm_proc_buf_disp[i], ssizes[target_rank], MPI_GRAPH_TYPE,
              target_rank, disp[target_rank], ssizes[target_rank], MPI_GRAPH_TYPE, commwin);
  }
#endif
#elif defined(USE_MPI_SENDRECV)
  for (int i = 0; i < nprocs; i++) {
      if (i != me)
          MPI_Sendrecv(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  gcomm, MPI_STATUSES_IGNORE);

      spos += ssizes[i];
      rpos += rsizes[i];
  }
#else
  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (rsizes[i] > 0))
      MPI_Irecv(rcdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, 
              CommunityTag, gcomm, &rreqs[i]);
    else
      rreqs[i] = MPI_REQUEST_NULL;

    rpos += rsizes[i];
  }
  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (ssizes[i] > 0))
      MPI_Isend(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, 
              CommunityTag, gcomm, &sreqs[i]);
    else
      sreqs[i] = MPI_REQUEST_NULL;

    spos += ssizes[i];
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

  // reserve vectors
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  for (GraphElem i = 0; i < nprocs; i++) {
      rcinfo[i].reserve(rpos);
  }
#endif

  // fetch baseptr from MPI window
#if defined(USE_MPI_RMA)
  MPI_Win_flush_all(commwin);
  MPI_Barrier(gcomm);

  GraphElem *rcbuf = nullptr;
  int flag = 0;
  MPI_Win_get_attr(commwin, MPI_WIN_BASE, &rcbuf, &flag);
#endif

  
  GraphElem ne = dg.get_ne();
  remoteComm.clear();
  remoteComm.resize(dg.get_nv(), -1); // the size of rvdata should be number of edges. it's possible we have missing comms stored here, so they have default -1


  for (GraphElem i = 0; i < rpos; i++) {

#if defined(USE_MPI_RMA)
    const GraphElem comm = rcbuf[i];
#else
    const GraphElem comm = rcdata[i];
#endif
    // rvdata has size of the incident vertices (rsz)
    remoteComm[rvdata[i]] = comm;
    const int tproc = dg.get_owner(comm);

    if (tproc != me)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
      rcinfo[tproc].emplace_back(comm);
#else
      rcinfo[tproc].insert(comm);
#endif
  }

  for (GraphElem i = 0; i < nv; i++) {
    const GraphElem comm = currComm[i];
    const int tproc = dg.get_owner(comm);

    if (tproc != me)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
      rcinfo[tproc].emplace_back(comm);
#else
      rcinfo[tproc].insert(comm);
#endif
  }

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  GraphElem stcsz = 0, rtcsz = 0;
  const int workGroupSize = getReductionWorkGroupSize(nprocs);
  // const int workGroupSize = 4;


  // Ported to SYCL
  GraphElem *_stcsz = sycl::malloc_shared<GraphElem>(1, q);
  *_stcsz = stcsz;

  auto _scsizes = scsizes.data();
  auto _rcinfo = rcinfo.data();

  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{nprocs, workGroupSize},
      sycl::reduction(_stcsz, std::plus<>()),
      [=](sycl::nd_item<1> it, auto &_stcsz){
      int i = it.get_global_id(0);
      _scsizes[i] = _rcinfo[i].size();
      _stcsz += _scsizes[i];
    });
  }).wait();

  stcsz = *_stcsz;
  sycl::free(_stcsz, q);
  // End SYCL port

  MPI_Alltoall(scsizes.data(), 1, MPI_GRAPH_TYPE, rcsizes.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif


  // SYCL Port
  auto _rcsizes = rcsizes.data();

  GraphElem *_rtcsz = sycl::malloc_shared<GraphElem>(1, q);
  *_rtcsz = rtcsz;

  // TODO: Replace explicit group size with default SYCL runtime group size selection
  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{nprocs, workGroupSize},
      sycl::reduction(_rtcsz, std::plus<>()),
      [=](sycl::nd_item<1> it, auto& _rtcsz) {
        int i = it.get_global_id(0);
        _rtcsz += _rcsizes[i];
    });
  }).wait();

  rtcsz = *_rtcsz;
  // SYCL Port End

#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]Total communities to receive: " << rtcsz << std::endl;
#endif
#if defined(USE_MPI_COLLECTIVES)
  std::vector<GraphElem, vec_ge_alloc> rcomms(rtcsz, vec_ge_allocator);
  std::vector<GraphElem> scomms(stcsz);
#else
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector<GraphElem, vec_ge_alloc> rcomms(rtcsz, vec_ge_allocator);
#else
  std::vector<GraphElem, vec_ge_alloc> rcomms(rtcsz, vec_ge_allocator);
  std::vector<GraphElem> scomms(stcsz);
#endif
#endif
  sinfo.resize(rtcsz);
  rinfo.resize(stcsz);

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  spos = 0;
  rpos = 0;
#if defined(USE_MPI_COLLECTIVES)
  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
          std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
      }
      scnts[i] = scsizes[i];
      rcnts[i] = rcsizes[i];
      sdispls[i] = spos;
      rdispls[i] = rpos;
      spos += scnts[i];
      rpos += rcnts[i];
  }
  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(scomms.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rcomms.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, gcomm);

  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
        // Porting to SYCL
        auto _rcomms = rcomms.data();
        auto _localCinfo = localCinfo.data();
        auto _sinfo = sinfo.data();
        auto _rdispls = rdispls.data();

        q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
          const int workGroupSize = getWorkGroupSize(rcsizes[i]);
          h.parallel_for(sycl::nd_range<1>{rcsizes[i], workGroupSize}, [=](sycl::nd_item<1> item){
            int j = item.get_global_id();
#else
          h.parallel_for(rcsizes[i], [=](sycl::id<1> j){
#endif
            const GraphElem comm = _rcomms[_rdispls[i] + j];
            _sinfo[_rdispls[i] + j] = {comm, _localCinfo[comm-base].size, _localCinfo[comm-base].degree};
          });
        }).wait();

      }
  }
  
  MPI_Alltoallv(sinfo.data(), rcnts.data(), rdispls.data(), 
          commType, rinfo.data(), scnts.data(), sdispls.data(), 
          commType, gcomm);
#else
#if !defined(USE_MPI_SENDRECV)
  std::vector<MPI_Request> rcreqs(nprocs);
#endif
  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#if defined(USE_MPI_SENDRECV)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
          MPI_Sendrecv(rcinfo[i].data(), scsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  gcomm, MPI_STATUSES_IGNORE);
#else
          std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
          MPI_Sendrecv(scomms.data() + spos, scsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  gcomm, MPI_STATUSES_IGNORE);
#endif
#else
          if (rcsizes[i] > 0) {
              MPI_Irecv(rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, 
                      CommunityTag, gcomm, &rreqs[i]);
          }
          else
              rreqs[i] = MPI_REQUEST_NULL;

          if (scsizes[i] > 0) {
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
              MPI_Isend(rcinfo[i].data(), scsizes[i], MPI_GRAPH_TYPE, i, 
                      CommunityTag, gcomm, &sreqs[i]);
#else
              std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
              MPI_Isend(scomms.data() + spos, scsizes[i], MPI_GRAPH_TYPE, i, 
                      CommunityTag, gcomm, &sreqs[i]);
#endif
          }
          else
              sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
  else {
#if !defined(USE_MPI_SENDRECV)
          rreqs[i] = MPI_REQUEST_NULL;
          sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      rpos += rcsizes[i];
      spos += scsizes[i];
  }

  spos = 0;
  rpos = 0;
          
  // poke progress on last isend/irecvs
#if !defined(USE_MPI_COLLECTIVES) && !defined(USE_MPI_SENDRECV) && defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
  int tf = 0, id = 0;
  MPI_Testany(nprocs, sreqs.data(), &id, &tf, MPI_STATUS_IGNORE);
#endif

#if !defined(USE_MPI_COLLECTIVES) && !defined(USE_MPI_SENDRECV) && !defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif

  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#if defined(USE_MPI_SENDRECV)
        // Porting to SYCL - Be careful with shadowing
        auto _rcomms = rcomms.data();
        auto _localCinfo = localCinfo.data();
        auto _sinfo = sinfo.data();

        q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
          const int workGroupSize = getWorkGroupSize(rcsizes[i]);
          h.parallel_for(sycl::nd_range<1>{rcsizes[i], workGroupSize}, [=](sycl::nd_item<1> item){
            int j = item.get_global_id();
#else
          h.parallel_for(rcsizes[i], [=](sycl::id<1> j){
#endif
            const GraphElem comm = _rcomms[rpos + j];
            _sinfo[rpos + j] = {comm, _localCinfo[comm-base].size, _localCinfo[comm-base].degree};
          });
        }).wait();

          
        MPI_Sendrecv(sinfo.data() + rpos, rcsizes[i], commType, i, CommunityDataTag, 
                rinfo.data() + spos, scsizes[i], commType, i, CommunityDataTag, 
                gcomm, MPI_STATUSES_IGNORE);
#else
          if (scsizes[i] > 0) {
              MPI_Irecv(rinfo.data() + spos, scsizes[i], commType, i, CommunityDataTag, 
                      gcomm, &rcreqs[i]);
          }
          else
              rcreqs[i] = MPI_REQUEST_NULL;

          // poke progress on last isend/irecvs
#if defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
          int flag = 0, done = 0;
          while (!done) {
              MPI_Test(&sreqs[i], &flag, MPI_STATUS_IGNORE);
              MPI_Test(&rreqs[i], &flag, MPI_STATUS_IGNORE);
              if (flag) 
                  done = 1;
          }
#endif

          // Porting to SYCL - Be careful with shadowing
          auto _rcomms = rcomms.data();
          auto _localCinfo = localCinfo.data();
          auto _sinfo = sinfo.data();

          q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
            const int workGroupSize = getWorkGroupSize(rcsizes[i]);
            h.parallel_for(sycl::nd_range<1>{rcsizes[i], workGroupSize}, [=](sycl::nd_item<1> item){
              int j = item.get_global_id();
#else
            h.parallel_for(rcsizes[i], [=](sycl::id<1> j){
#endif
              const GraphElem comm = _rcomms[rpos + j];
              _sinfo[rpos + j] = {comm, _localCinfo[comm-base].size, _localCinfo[comm-base].degree};
            });
          }).wait();

          // End of port

          if (rcsizes[i] > 0) {
              MPI_Isend(sinfo.data() + rpos, rcsizes[i], commType, i, 
                      CommunityDataTag, gcomm, &sreqs[i]);
          }
          else
              sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      else {
#if !defined(USE_MPI_SENDRECV)
          rcreqs[i] = MPI_REQUEST_NULL;
          sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      rpos += rcsizes[i];
      spos += scsizes[i];
  }

#if !defined(USE_MPI_SENDRECV)
  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rcreqs.data(), MPI_STATUSES_IGNORE);
#endif

#endif

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

  remoteCinfo.clear();
  remoteCupdate.clear();
  remoteCinfo.resize(dg.get_nv(), Comm(-1, 0.0));
  remoteCupdate.resize(dg.get_nv(), Comm(-1, 0.0));

  for (GraphElem i = 0; i < stcsz; i++) {
      const GraphElem ccomm = rinfo[i].community;

      Comm comm;

      comm.size = rinfo[i].size;
      comm.degree = rinfo[i].degree;

      remoteCinfo[ccomm] = comm;
      remoteCupdate[ccomm] = Comm();
  }
} // end fillRemoteCommunities

void createCommunityMPIType()
{
  CommInfo cinfo;

  MPI_Aint begin, community, size, degree;

  MPI_Get_address(&cinfo, &begin);
  MPI_Get_address(&cinfo.community, &community);
  MPI_Get_address(&cinfo.size, &size);
  MPI_Get_address(&cinfo.degree, &degree);

  int blens[] = { 1, 1, 1 };
  MPI_Aint displ[] = { community - begin, size - begin, degree - begin };
  MPI_Datatype types[] = { MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_WEIGHT_TYPE };

  MPI_Type_create_struct(3, blens, displ, types, &commType);
  MPI_Type_commit(&commType);
} // createCommunityMPIType

void destroyCommunityMPIType()
{
  MPI_Type_free(&commType);
} // destroyCommunityMPIType

void updateRemoteCommunities(const Graph &dg, std::vector<Comm, vec_comm_alloc> &localCinfo,
			                      const std::vector<Comm, vec_comm_alloc> &remoteCupdate,
			                      const int me, const int nprocs)
{
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  std::vector<std::vector<CommInfo>> remoteArray(nprocs);
  MPI_Comm gcomm = dg.get_comm();
  
  // TODO: Parallelize this (authors notes)
  int counter = 0;
  for (auto iter = remoteCupdate.begin(); iter != remoteCupdate.end(); iter++) {
      const GraphElem i = counter;
      const Comm &curr = *iter;
      
      // if an invalid community, then the comm.size is set to -1
      if (curr.size == -1) continue;

      const int tproc = dg.get_owner(i);

#ifdef DEBUG_ASSERTIONS  
      assert(tproc != me);
#endif
      CommInfo rcinfo;

      rcinfo.community = i;
      rcinfo.size = curr.size;
      rcinfo.degree = curr.degree;

      remoteArray[tproc].push_back(rcinfo);
  }

  std::vector<GraphElem> send_sz(nprocs), recv_sz(nprocs);

#ifdef DEBUG_PRINTF  
  GraphWeight tc = 0.0;
  const double t0 = MPI_Wtime();
#endif

  {
    // Ported to SYCL
    auto _send_sz = send_sz.data();
    auto _remoteArray = remoteArray.data();


    q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
      const int workGroupSize = getWorkGroupSize(nprocs);
      h.parallel_for(sycl::nd_range<1>{nprocs, workGroupSize}, [=](sycl::nd_item<1> item){
        int i = item.get_global_id();
#else
      h.parallel_for(nprocs, [=](sycl::id<1> i){
#endif
        _send_sz[i] = _remoteArray[i].size();
      });
    }).wait();
  }

  MPI_Alltoall(send_sz.data(), 1, MPI_GRAPH_TYPE, recv_sz.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

#ifdef DEBUG_PRINTF  
  const double t1 = MPI_Wtime();
  tc += (t1 - t0);
#endif

  GraphElem rcnt = 0, scnt = 0;
  GraphElem *_rcnt = sycl::malloc_shared<GraphElem>(1, q);
  GraphElem *_scnt = sycl::malloc_shared<GraphElem>(1, q);
  
  *_scnt = scnt;
  *_rcnt = rcnt;


  // Ported to SYCL
  {
    // NOTE: I've been unable to combine the below two into a
    // double reduction without getting runtime errors
    const int workGroupSize = getReductionWorkGroupSize(nprocs);
    // const int workGroupSize = 4;

    auto _send_sz = send_sz.data();
    auto _recv_sz = recv_sz.data();

    q.submit([&](sycl::handler &h){
      h.parallel_for(
          sycl::nd_range<1>{nprocs, workGroupSize},
          sycl::reduction(_rcnt, std::plus<>()),
          [=](sycl::nd_item<1> it, auto& _rcnt) {
            int i = it.get_global_id(0);
            _rcnt += _recv_sz[i];
      });
    });

    q.submit([&](sycl::handler &h){
      h.parallel_for(
          sycl::nd_range<1>{nprocs, workGroupSize},
          sycl::reduction(_scnt, std::plus<>()),
          [=](sycl::nd_item<1> it, auto& _scnt) {
            int i = it.get_global_id(0);
            _scnt += _send_sz[i];
      });
    });

    q.wait();
  }


  scnt = *_scnt;
  rcnt = *_rcnt;

  // End of Port

#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]Total number of remote communities to update: " << scnt << std::endl;
#endif

  GraphElem currPos = 0;
  std::vector<CommInfo> rdata(rcnt);

#ifdef DEBUG_PRINTF  
  const double t2 = MPI_Wtime();
#endif
#if defined(USE_MPI_SENDRECV)
  for (int i = 0; i < nprocs; i++) {
      if (i != me)
          MPI_Sendrecv(remoteArray[i].data(), send_sz[i], commType, i, CommunityDataTag, 
                  rdata.data() + currPos, recv_sz[i], commType, i, CommunityDataTag, 
                  gcomm, MPI_STATUSES_IGNORE);

      currPos += recv_sz[i];
  }
#else
  std::vector<MPI_Request> sreqs(nprocs), rreqs(nprocs);
  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (recv_sz[i] > 0))
      MPI_Irecv(rdata.data() + currPos, recv_sz[i], commType, i, 
              CommunityDataTag, gcomm, &rreqs[i]);
    else
      rreqs[i] = MPI_REQUEST_NULL;

    currPos += recv_sz[i];
  }

  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (send_sz[i] > 0))
      MPI_Isend(remoteArray[i].data(), send_sz[i], commType, i, 
              CommunityDataTag, gcomm, &sreqs[i]);
    else
      sreqs[i] = MPI_REQUEST_NULL;
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif
#ifdef DEBUG_PRINTF  
  const double t3 = MPI_Wtime();
  std::cout << "[" << me << "]Update remote community MPI time: " << (t3 - t2) << std::endl;
#endif


  {
    // Ported to SYCL
    auto _localCinfo = localCinfo.data();
    auto _rdata = rdata.data();

    q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
      const int workGroupSize = getWorkGroupSize(rcnt);
      h.parallel_for(sycl::nd_range<1>{rcnt, workGroupSize}, [=](sycl::nd_item<1> item){
        int i = item.get_global_id();
#else
      h.parallel_for(rcnt, [=](sycl::id<1> i){
#endif
        const CommInfo &curr = _rdata[i];
    #ifdef DEBUG_ASSERTIONS  
        assert(dg.get_owner(curr.community) == me);
    #endif
        _localCinfo[curr.community-base].size += curr.size;
        _localCinfo[curr.community-base].degree += curr.degree;
      });
    }).wait();
  }


} // updateRemoteCommunities

// initial setup before Louvain iteration begins
#if defined(USE_MPI_RMA)
void exchangeVertexReqs(const Graph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem, vec_ge_alloc> &rsizes, 
        std::vector<GraphElem, vec_ge_alloc> &svdata, std::vector<GraphElem, vec_ge_alloc> &rvdata,
        const int me, const int nprocs, MPI_Win &commwin)
#else
void exchangeVertexReqs(const Graph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem, vec_ge_alloc> &rsizes, 
        std::vector<GraphElem, vec_ge_alloc> &svdata, std::vector<GraphElem, vec_ge_alloc> &rvdata,
        const int me, const int nprocs)
#endif
{
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  std::vector<std::unordered_set<GraphElem>> parray(nprocs);

  // NOTE: Did not port the below into SYCL
  for (GraphElem i = 0; i < nv; i++) {
    GraphElem e0, e1;

    dg.edge_range(i, e0, e1);

    for (GraphElem j = e0; j < e1; j++) {
      const Edge &edge = dg.get_edge(j);
      const int tproc = dg.get_owner(edge.tail_);

      if (tproc != me) {
        parray[tproc].insert(edge.tail_);
      }
    }
  }


  rsizes.resize(nprocs);
  ssizes.resize(nprocs);
  ssz = 0, rsz = 0;

  int pproc = 0;
  // TODO FIXME parallelize this loop (original authors)
  for (std::vector<std::unordered_set<GraphElem>>::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
    ssz += iter->size();
    ssizes[pproc] = iter->size();
    pproc++;
  }

  MPI_Alltoall(ssizes.data(), 1, MPI_GRAPH_TYPE, rsizes.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

  GraphElem rsz_r = 0;
  
  // SYCL Port
  GraphElem *_rsz_r = sycl::malloc_shared<GraphElem>(1, q);
  *_rsz_r = rsz_r;
  auto _rsizes = rsizes.data();

  const int workGroupSize = getReductionWorkGroupSize(nprocs);
  // const int workGroupSize = 4;

  q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::nd_range<1> {nprocs, workGroupSize},
                  sycl::reduction(_rsz_r, std::plus<>()),
                  [=](sycl::nd_item<1> it, auto &_rsz_r){
                    int i = it.get_global_id(0);
                    _rsz_r += _rsizes[i];
                  });
  }).wait();

  rsz_r = *_rsz_r;
  rsz = rsz_r;
  // SYCL Port end

  svdata.resize(ssz);
  rvdata.resize(rsz);

  GraphElem cpos = 0, rpos = 0;
  pproc = 0;

#if defined(USE_MPI_COLLECTIVES)
  std::vector<int> scnts(nprocs), rcnts(nprocs), sdispls(nprocs), rdispls(nprocs);
  
  for (std::vector<std::unordered_set<GraphElem>>::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
      std::copy(iter->begin(), iter->end(), svdata.begin() + cpos);
      
      scnts[pproc] = iter->size();
      rcnts[pproc] = rsizes[pproc];
      sdispls[pproc] = cpos;
      rdispls[pproc] = rpos;
      cpos += iter->size();
      rpos += rcnts[pproc];

      pproc++;
  }

  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(svdata.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rvdata.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, gcomm);
#else
  std::vector<MPI_Request> rreqs(nprocs), sreqs(nprocs);
  for (int i = 0; i < nprocs; i++) {
      if ((i != me) && (rsizes[i] > 0))
          MPI_Irecv(rvdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, 
                  VertexTag, gcomm, &rreqs[i]);
      else
          rreqs[i] = MPI_REQUEST_NULL;

      rpos += rsizes[i];
  }

  for (std::vector<std::unordered_set<GraphElem>>::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
      std::copy(iter->begin(), iter->end(), svdata.begin() + cpos);

      if ((me != pproc) && (iter->size() > 0))
          MPI_Isend(svdata.data() + cpos, iter->size(), MPI_GRAPH_TYPE, pproc, 
                  VertexTag, gcomm, &sreqs[pproc]);
      else
          sreqs[pproc] = MPI_REQUEST_NULL;

      cpos += iter->size();
      pproc++;
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif

  std::swap(svdata, rvdata);
  
  // Cannot perform std::swap on vectors with different allocators 
  // (i.e. C++ default alloc vs USM shared memory allocator)
  std::vector<GraphElem> temp(ssizes.begin(), ssizes.end());
  ssizes.assign(rsizes.begin(), rsizes.end());
  rsizes.assign(temp.begin(), temp.end());

  std::swap(ssz, rsz);

  // create MPI window for communities
#if defined(USE_MPI_RMA)  
  GraphElem *ptr = nullptr;
  MPI_Info info = MPI_INFO_NULL;
#if defined(USE_MPI_ACCUMULATE)
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Info_set(info, "accumulate_ops", "same_op");
#endif
  MPI_Win_allocate(rsz*sizeof(GraphElem), sizeof(GraphElem), 
          info, gcomm, &ptr, &commwin);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, commwin);
#endif
} // exchangeVertexReqs

#if defined(USE_MPI_RMA)
GraphWeight distLouvainMethod(const int me, const int nprocs, const Graph &dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, std::vector<GraphElem> &_rsizes, 
        std::vector<GraphElem> &_svdata, std::vector<GraphElem> &rvdata, const GraphWeight lower, 
        const GraphWeight thresh, int &iters, MPI_Win &commwin)
#else
GraphWeight distLouvainMethod(const int me, const int nprocs, const Graph &_dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, std::vector<GraphElem> &_rsizes, 
        std::vector<GraphElem> &_svdata, std::vector<GraphElem> &_rvdata, const GraphWeight lower, 
        const GraphWeight thresh, int &iters)
#endif
{
  std::vector<GraphWeight, vec_gw_alloc> vDegree(vec_gw_allocator);
  std::vector<GraphElem, vec_ge_alloc> pastComm(vec_ge_allocator), currComm(vec_ge_allocator), targetComm(vec_ge_allocator);
  std::vector<Comm, vec_comm_alloc> localCinfo(vec_comm_allocator), localCupdate(vec_comm_allocator);
  std::vector<GraphWeight, vec_gw_alloc> clusterWeight(vec_gw_allocator);
  std::vector<GraphElem, vec_ge_alloc> remoteComm(vec_ge_allocator);
  std::vector<Comm, vec_comm_alloc> remoteCinfo(vec_comm_allocator), remoteCupdate(vec_comm_allocator);

  void *memory_block = sycl::malloc_shared<Graph>(1, q);
  const Graph dg = *new(memory_block) Graph(_dg);

  std::vector<GraphElem, vec_ge_alloc> rsizes(_rsizes.begin(), _rsizes.end(), vec_ge_allocator);
  std::vector<GraphElem, vec_ge_alloc> svdata(_svdata.begin(), _svdata.end(), vec_ge_allocator);
  std::vector<GraphElem, vec_ge_alloc> rvdata(_rvdata.begin(), _rvdata.end(), vec_ge_allocator);
  
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;
  
  // Ported to SYCL
  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);


#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]constantForSecondTerm: " << constantForSecondTerm << std::endl;
  if (me == 0)
      std::cout << "Threshold: " << thresh << std::endl;
#endif
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);

#ifdef DEBUG_PRINTF  
  double t0, t1;
  t0 = MPI_Wtime();
#endif

  // setup vertices and communities
#if defined(USE_MPI_RMA)
  // Ported to SYCL
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs, commwin);
  
  // store the remote displacements 
  std::vector<GraphElem> disp(nprocs);
  MPI_Exscan(ssizes.data(), (GraphElem*)disp.data(), nprocs, MPI_GRAPH_TYPE, 
          MPI_SUM, gcomm);
#else
  // Ported to SYCL
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#endif

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  std::cout << "[" << me << "]Initial communication setup time before Louvain iteration (in s): " << (t1 - t0) << std::endl;
#endif

 
  // start Louvain iteration
  while(true) {
#ifdef DEBUG_PRINTF  
    const double t2 = MPI_Wtime();
    if (me == 0)
        std::cout << "Starting Louvain iteration: " << numIters << std::endl;
#endif
    numIters++;

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

#if defined(USE_MPI_RMA)
    // Ported to SYCL
    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate, 
            commwin, disp);
#else
    // Ported to SYCL
    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate);
#endif


#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    int remoteCommSize = 0;
    for (auto it = remoteComm.begin(); it != remoteComm.end(); it++){
      if (*it != -1)
        remoteCommSize += 1;
    }
    std::cout << "[" << me << "]Remote community map size: " << remoteComm.size() << std::endl;
    std::cout << "[" << me << "]Iteration communication time: " << (t1 - t0) << std::endl;
#endif

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

    // Ported to SYCL
    distCleanCWandCU(nv, clusterWeight, localCupdate);

    // Ported to SYCL
    distExecuteLouvainIteration(nv, dg, currComm, targetComm, vDegree, localCinfo, 
                                    localCupdate, remoteComm, remoteCinfo, remoteCupdate,
                                    constantForSecondTerm, clusterWeight, me);

    // Ported to SYCL
    distUpdateLocalCinfo(localCinfo, localCupdate);

    // Ported to SYCL
    updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);
    
    // Ported to SYCL
    currMod = distComputeModularity(dg, localCinfo, clusterWeight, constantForSecondTerm, me);

    // exit criteria
    if (currMod - prevMod < thresh)
        break;

    prevMod = currMod;
    if (prevMod < lower)
        prevMod = lower;

    // Ported to SYCL
    {
      auto _pastComm = pastComm.data();
      auto _currComm = currComm.data();
      auto _targetComm = targetComm.data();

      q.submit([&](sycl::handler &h){
#ifdef SCALING_TESTS
        const int workGroupSize = getWorkGroupSize(nv);
        h.parallel_for(sycl::nd_range<1>{nv, workGroupSize}, [=](sycl::nd_item<1> item){
          int i = item.get_global_id();
#else
        h.parallel_for(nv, [=](sycl::id<1> i){
#endif
          GraphElem tmp = _pastComm[i];
          _pastComm[i] = _currComm[i];
          _currComm[i] = _targetComm[i];
          _targetComm[i] = tmp;
        });
      }).wait();
    }
    
  } // end of Louvain iteration

#if defined(USE_MPI_RMA)
  MPI_Win_unlock_all(commwin);
  MPI_Win_free(&commwin);
#endif  

  iters = numIters;

  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  // we then need to free the allocated block
  sycl::free(memory_block, q);

  return prevMod;
} // distLouvainMethod plain

#endif // __DSPL
