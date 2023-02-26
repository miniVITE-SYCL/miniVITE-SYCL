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
#include <omp.h>
#include <CL/sycl.hpp>
#include "graph.hpp"
#include "utils.hpp"

struct Comm {
  GraphElem size;
  GraphWeight degree;

  Comm() : size(0), degree(0.0) {};
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
typedef sycl::usm_allocator<std::vector<CommInfo>, sycl::usm::alloc::shared> vec_vec_commi_alloc;
typedef sycl::usm_allocator<int, sycl::usm::alloc::shared> vec_int_alloc;
typedef sycl::usm_allocator<vec_ge_alloc, sycl::usm::alloc::shared> vec_vec_ge_alloc;
typedef sycl::usm_allocator<vec_bool_alloc, sycl::usm::alloc::shared> vec_vec_bool_alloc;
typedef sycl::usm_allocator<std::unordered_set<GraphElem>, sycl::usm::alloc::shared> vec_uset_ge_alloc;
// typedef sycl::usm_allocator<std::unordered_set<GraphElem, vec_ge_alloc>, sycl::usm::alloc::shared> vec_uset_ge_alloc;

// Defined a SYCL queue using the CPU selector
sycl::queue q{sycl::cpu_selector{}};

// we instantiate USM STL allocators (dependency on sycl::queue q)
vec_gw_alloc vec_gw_allocator(q);
vec_ge_alloc vec_ge_allocator(q);
vec_comm_alloc vec_comm_allocator(q);
vec_commi_alloc vec_commi_allocator(q);
vec_vec_commi_alloc vec_vec_commi_allocator(q);
vec_int_alloc vec_int_allocator(q);
vec_vec_ge_alloc vec_vec_ge_allocator(q);
vec_uset_ge_alloc vec_uset_ge_allocator(q);
vec_vec_bool_alloc vec_vec_bool_allocator(q);


void distSumVertexDegree(const Graph &g, std::vector<GraphWeight, vec_gw_alloc> &vDegree, std::vector<Comm, vec_comm_alloc> &localCinfo, sycl::queue &q)
{

  // we then create pointers to the underlying data
  auto _vDegree = vDegree.data();
  auto _localCinfo = localCinfo.data();
  const Graph *_g = &g;

  const GraphElem nv = g.get_lnv();

  q.submit([&](sycl::handler &h){
    h.parallel_for(nv, [=](sycl::id<1> i){
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
  }).wait(); // currently we wait until this has been executed
  // Omp parallel blocks have barriers at the end so this is the same


}


GraphWeight distCalcConstantForSecondTerm(const std::vector<GraphWeight, vec_gw_alloc> &vDegree, MPI_Comm gcomm, sycl::queue &q)
{

  GraphWeight totalEdgeWeightTwice = 0.0;
  int me = -1;

  const size_t vsz = vDegree.size();
  const int local_group_size = 4;

  // TODO: Why do we need to use USM?? why can't we copy this value in?
  // NOTE: I got the idea to do this when seeing some slides that performed
  // reduction using buffers (so I subbed out USM)
  // we then create pointers to the underlying data
  auto _vDegree = vDegree.data();
  GraphWeight localWeight = 0;
  GraphWeight *usm_localWeight = sycl::malloc_host<GraphWeight>(1, q);
  *usm_localWeight = 0;

  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{vsz, local_group_size},
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

#ifdef DEBUG_PRINTF  
  assert(csz == pastComm.size());
#endif

  auto _pastComm = pastComm.data();
  auto _currComm = currComm.data();

  q.submit([&](sycl::handler &h){
    h.parallel_for(csz, [=](sycl::id<1> i){
      _pastComm[i] = i + base;
      _currComm[i] = i + base;
    });
  }).wait();

} // distInitComm

void distInitLouvain(const Graph &dg, std::vector<GraphElem> &pastComm, 
        std::vector<GraphElem> &currComm, std::vector<GraphWeight> &vDegree, 
        std::vector<GraphWeight> &clusterWeight, std::vector<Comm> &localCinfo, 
        std::vector<Comm> &localCupdate, GraphWeight &constantForSecondTerm,
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

  // SYCL Construct and USM handling Start --------
  // we use the allocated memory in the constructor
  std::vector<GraphWeight, vec_gw_alloc> usm_vDegree(vDegree.size(), vec_gw_allocator);
  std::vector<GraphElem, vec_ge_alloc> usm_pastComm(pastComm.size(), vec_ge_allocator),
                                       usm_currComm(currComm.size(), vec_ge_allocator);
  std::vector<Comm, vec_comm_alloc> usm_localCinfo(localCinfo.size(), vec_comm_allocator);
  // we create a shadow usm memory graph
  void *memory_block = sycl::malloc_shared<Graph>(1, q);
  Graph *usm_g = new(memory_block) Graph(dg);
  // SYCL Construct and USM handling End --------

  distSumVertexDegree(*usm_g, usm_vDegree, usm_localCinfo, q);
  constantForSecondTerm = distCalcConstantForSecondTerm(usm_vDegree, gcomm, q);
  distInitComm(usm_pastComm, usm_currComm, base, q);
  
  // SYCL Construct and USM Destruction start ---------
  // update our original STL containers from our SYCL USM memory
  std::memcpy(vDegree.data(), usm_vDegree.data(), vDegree.size() * sizeof(GraphWeight));
  std::memcpy(localCinfo.data(), usm_localCinfo.data(), localCinfo.size() * sizeof(Comm));
  std::memcpy(currComm.data(), usm_currComm.data(), currComm.size() * sizeof(GraphElem));
  std::memcpy(pastComm.data(), usm_pastComm.data(), pastComm.size() * sizeof(GraphElem));
  // We need to call the deconstructor for new placement objects
  usm_g->~Graph();
  // we then need to free the allocated block
  sycl::free(usm_g, q);
  // SYCL Construct and USM Destruction End ---------

} // distInitLouvain

GraphElem distGetMaxIndex(const std::unordered_map<GraphElem, GraphElem> &clmap, const std::vector<GraphWeight> &counter,
			  const GraphWeight selfLoop, const std::vector<Comm> &localCinfo, 
			  const std::map<GraphElem,Comm> &remoteCinfo, const GraphWeight vDegree, 
                          const GraphElem currSize, const GraphWeight currDegree, const GraphElem currComm,
			  const GraphElem base, const GraphElem bound, const GraphWeight constant)
{
  std::unordered_map<GraphElem, GraphElem>::const_iterator storedAlready;
  GraphElem maxIndex = currComm;
  GraphWeight curGain = 0.0, maxGain = 0.0;
  GraphWeight eix = static_cast<GraphWeight>(counter[0]) - static_cast<GraphWeight>(selfLoop);

  GraphWeight ax = currDegree - vDegree;
  GraphWeight eiy = 0.0, ay = 0.0;

  GraphElem maxSize = currSize; 
  GraphElem size = 0;

  storedAlready = clmap.begin();
#ifdef DEBUG_PRINTF  
  assert(storedAlready != clmap.end());
#endif
  do {
      if (currComm != storedAlready->first) {

          // is_local, direct access local info
          if ((storedAlready->first >= base) && (storedAlready->first < bound)) {
              ay = localCinfo[storedAlready->first-base].degree;
              size = localCinfo[storedAlready->first - base].size;   
          }
          else {
              // is_remote, lookup map
              std::map<GraphElem,Comm>::const_iterator citer = remoteCinfo.find(storedAlready->first);
              ay = citer->second.degree;
              size = citer->second.size; 
          }

          eiy = counter[storedAlready->second];

          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant;

          if ((curGain > maxGain) ||
                  ((curGain == maxGain) && (curGain != 0.0) && (storedAlready->first < maxIndex))) {
              maxGain = curGain;
              maxIndex = storedAlready->first;
              maxSize = size;
          }
      }
      storedAlready++;
  } while (storedAlready != clmap.end());

  if ((maxSize == 1) && (currSize == 1) && (maxIndex > currComm))
    maxIndex = currComm;

  return maxIndex;
} // distGetMaxIndex

GraphWeight distBuildLocalMapCounter(const GraphElem e0, const GraphElem e1, std::unordered_map<GraphElem, GraphElem> &clmap, 
				   std::vector<GraphWeight> &counter, const Graph &g, 
                                   const std::vector<GraphElem> &currComm, 
                                   const std::unordered_map<GraphElem, GraphElem> &remoteComm,
	                           const GraphElem vertex, const GraphElem base, const GraphElem bound)
{
  GraphElem numUniqueClusters = 1L;
  GraphWeight selfLoop = 0;
  std::unordered_map<GraphElem, GraphElem>::const_iterator storedAlready;

  for (GraphElem j = e0; j < e1; j++) {
        
    const Edge &edge = g.get_edge(j);
    const GraphElem &tail_ = edge.tail_;
    const GraphWeight &weight = edge.weight_;
    GraphElem tcomm;

    if (tail_ == vertex + base)
      selfLoop += weight;

    // is_local, direct access local std::vector<GraphElem>
    if ((tail_ >= base) && (tail_ < bound))
      tcomm = currComm[tail_ - base];
    else { // is_remote, lookup map
      std::unordered_map<GraphElem, GraphElem>::const_iterator iter = remoteComm.find(tail_);

#ifdef DEBUG_PRINTF  
      assert(iter != remoteComm.end());
#endif
      tcomm = iter->second;
    }

    storedAlready = clmap.find(tcomm);
    
    if (storedAlready != clmap.end())
      counter[storedAlready->second] += weight;
    else {
        clmap.insert(std::unordered_map<GraphElem, GraphElem>::value_type(tcomm, numUniqueClusters));
        counter.push_back(weight);
        numUniqueClusters++;
    }
  }

  return selfLoop;
} // distBuildLocalMapCounter

void distExecuteLouvainIteration(const GraphElem i, const Graph &dg, const std::vector<GraphElem> &currComm,
				 std::vector<GraphElem> &targetComm, const std::vector<GraphWeight> &vDegree,
                                 std::vector<Comm> &localCinfo, std::vector<Comm> &localCupdate,
				 const std::unordered_map<GraphElem, GraphElem> &remoteComm, 
                                 const std::map<GraphElem,Comm> &remoteCinfo, 
                                 std::map<GraphElem,Comm> &remoteCupdate, const GraphWeight constantForSecondTerm,
                                 std::vector<GraphWeight> &clusterWeight, const int me)
{
  // std::cout << "Dist Louvain Method Iteration" << std::endl;
  GraphElem localTarget = -1;
  GraphElem e0, e1, selfLoop = 0;
  std::unordered_map<GraphElem, GraphElem> clmap;
  std::vector<GraphWeight> counter;

  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  const GraphElem cc = currComm[i];
  GraphWeight ccDegree;
  GraphElem ccSize;  
  bool currCommIsLocal = false; 
  bool targetCommIsLocal = false;

  // Current Community is local
  if (cc >= base && cc < bound) {
	  ccDegree=localCinfo[cc-base].degree;
    ccSize=localCinfo[cc-base].size;
    currCommIsLocal=true;
  } else {
  // is remote
    std::map<GraphElem,Comm>::const_iterator citer = remoteCinfo.find(cc);
	  ccDegree = citer->second.degree;
 	  ccSize = citer->second.size;
	  currCommIsLocal=false;
  }

  dg.edge_range(i, e0, e1);

  if (e0 != e1) {
    clmap.insert(std::unordered_map<GraphElem, GraphElem>::value_type(cc, 0));
    counter.push_back(0.0);

    selfLoop =  distBuildLocalMapCounter(e0, e1, clmap, counter, dg, 
                    currComm, remoteComm, i, base, bound);

    clusterWeight[i] += counter[0];

    localTarget = distGetMaxIndex(clmap, counter, selfLoop, localCinfo, remoteCinfo, 
                    vDegree[i], ccSize, ccDegree, cc, base, bound, constantForSecondTerm);
  }
  else
    localTarget = cc;

   // is the Target Local?
   if (localTarget >= base && localTarget < bound)
      targetCommIsLocal = true;
  
  // current and target comm are local - atomic updates to vectors
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && targetCommIsLocal) {
        
#ifdef DEBUG_PRINTF  
    assert( base < localTarget < bound);
    assert( base < cc < bound);
	  assert( cc - base < localCupdate.size()); 	
	  assert( localTarget - base < localCupdate.size()); 	
#endif
    #pragma omp atomic update
    localCupdate[localTarget-base].degree += vDegree[i];
    #pragma omp atomic update
    localCupdate[localTarget-base].size++;
    #pragma omp atomic update
    localCupdate[cc-base].degree -= vDegree[i];
    #pragma omp atomic update
    localCupdate[cc-base].size--;
  }	

  // current is local, target is not - do atomic on local, accumulate in Maps for remote
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && !targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[cc-base].degree -= vDegree[i];
        #pragma omp atomic update
        localCupdate[cc-base].size--;
 
        // search target!     
        std::map<GraphElem,Comm>::iterator iter=remoteCupdate.find(localTarget);
 
        #pragma omp atomic update
        iter->second.degree += vDegree[i];
        #pragma omp atomic update
        iter->second.size++;
  }
        
   // current is remote, target is local - accumulate for current, atomic on local
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[localTarget-base].degree += vDegree[i];
        #pragma omp atomic update
        localCupdate[localTarget-base].size++;
       
        // search current 
        std::map<GraphElem,Comm>::iterator iter=remoteCupdate.find(cc);
  
        #pragma omp atomic update
        iter->second.degree -= vDegree[i];
        #pragma omp atomic update
        iter->second.size--;
   }
                    
   // current and target are remote - accumulate for both
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && !targetCommIsLocal) {
       
        // search current 
        std::map<GraphElem,Comm>::iterator iter = remoteCupdate.find(cc);
  
        #pragma omp atomic update
        iter->second.degree -= vDegree[i];
        #pragma omp atomic update
        iter->second.size--;
   
        // search target
        iter=remoteCupdate.find(localTarget);
  
        #pragma omp atomic update
        iter->second.degree += vDegree[i];
        #pragma omp atomic update
        iter->second.size++;
   }

#ifdef DEBUG_PRINTF  
  assert(localTarget != -1);
#endif
  targetComm[i] = localTarget;
} // distExecuteLouvainIteration

GraphWeight distComputeModularity(const Graph &g, std::vector<Comm> &localCinfo,
			     const std::vector<GraphWeight> &clusterWeight,
			     const GraphWeight constantForSecondTerm,
			     const int me)
{
  const GraphElem nv = g.get_lnv();
  MPI_Comm gcomm = g.get_comm();

  GraphWeight le_la_xx[2];
  GraphWeight e_a_xx[2] = {0.0, 0.0};
  GraphWeight le_xx = 0.0, la2_x = 0.0;

#ifdef DEBUG_PRINTF  
  assert((clusterWeight.size() == nv));
#endif

  // auto _accumulator = sycl::malloc_host<GraphWeight>(2, q);
  std::vector<Comm, vec_comm_alloc> usm_localCinfo(localCinfo.begin(), localCinfo.end(), vec_comm_allocator);
  std::vector<GraphWeight, vec_gw_alloc> usm_clusterWeight(clusterWeight.begin(), clusterWeight.end(), vec_gw_allocator);
  auto _localCinfo = usm_localCinfo.data();
  auto _clusterWeight = usm_clusterWeight.data();

  GraphWeight *_le_xx = sycl::malloc_host<GraphWeight>(1, q);
  GraphWeight *_la2_x = sycl::malloc_host<GraphWeight>(1, q);
  *_le_xx = le_xx;
  *_la2_x = la2_x;

  // NOTE: The order of the arguments matters for the parallel_for lambda
  // This order corresponds to the order of the reductions

  int local_group_size = 4;
  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{nv, local_group_size},
      sycl::reduction(_le_xx, std::plus<>()),
      [=](sycl::nd_item<1> it, auto &_le_xx){
        int i = it.get_global_id(0);
        _le_xx += _clusterWeight[i];
    });
  });

  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{nv, local_group_size},
      sycl::reduction(_la2_x, std::plus<>()),
      [=](sycl::nd_item<1> it, auto &_la2_x){
        int i = it.get_global_id(0);
        _la2_x += static_cast<GraphWeight>(_localCinfo[i].degree) * static_cast<GraphWeight>(_localCinfo[i].degree); 
    });
  });

  q.wait();

  // BUG: The double reduction below doesn't work for some reason?
  // Reproduce issue -> mpirun -n 4 ./miniVITE-SYCL -n 100
  // This doesn't manifest with MPI np=1, or another value (I think)
  // This issues doesn't manifest when local_group_size=1 (only value tested)
  // ==> It's possible that the issue still exists in the above fix attempt, but doesn't manifest in that specific program scenario

  // q.submit([&](sycl::handler &h){
  //   h.parallel_for(sycl::nd_range<1>{nv, local_group_size},
  //                  sycl::reduction(_le_xx, std::plus<>()),
  //                  sycl::reduction(_la2_x, std::plus<>()),
  //                  [=](sycl::nd_item<1> it, auto &_le_xx, auto &_la2_x){
  //                     int i = it.get_global_id(0);
  //                     _le_xx += _clusterWeight[i];
  //                     // BUG: localCinfo is of type Comm, why are we casting to GraphWeight? Is this the correct thing
  //                     _la2_x += static_cast<GraphWeight>(_localCinfo[i].degree) * static_cast<GraphWeight>(_localCinfo[i].degree); 
  //                  });
  // }).wait();

  q.wait();

  le_xx = *_le_xx;
  la2_x = *_la2_x;

  le_la_xx[0] = le_xx;
  le_la_xx[1] = la2_x;

  sycl::free(_le_xx, q);
  sycl::free(_la2_x, q);

#ifdef DEBUG_PRINTF  
  const double t0 = MPI_Wtime();
#endif

  MPI_Allreduce(le_la_xx, e_a_xx, 2, MPI_WEIGHT_TYPE, MPI_SUM, gcomm);

#ifdef DEBUG_PRINTF  
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

void distUpdateLocalCinfo(std::vector<Comm> &localCinfo, const std::vector<Comm> &localCupdate)
{
  size_t csz = localCinfo.size();

  std::vector<Comm, vec_comm_alloc> usm_localCinfo(localCinfo.begin(), localCinfo.end(), vec_comm_allocator);
  std::vector<Comm, vec_comm_alloc> usm_localCupdate(localCupdate.begin(), localCupdate.end(), vec_comm_allocator);

  auto _localCinfo = usm_localCinfo.data();
  auto _localCupdate = usm_localCupdate.data();

  q.submit([&](sycl::handler &h){
    h.parallel_for(csz, [=](sycl::id<1> i){
      _localCinfo[i].size += _localCupdate[i].size;
      _localCinfo[i].degree += _localCupdate[i].degree;
    });
  }).wait();

  std::memcpy(localCinfo.data(), usm_localCinfo.data(), usm_localCinfo.size() * sizeof(Comm));
  // localCupdate is not updated -- check const above
}

void distCleanCWandCU(const GraphElem nv, std::vector<GraphWeight> &clusterWeight,
        std::vector<Comm> &localCupdate)
{
  // We provide local SYCL USM constructs
  std::vector<GraphWeight, vec_gw_alloc> usm_clusterWeight(clusterWeight.begin(), clusterWeight.end(), vec_gw_allocator);
  std::vector<Comm, vec_comm_alloc> usm_localCupdate(localCupdate.begin(), localCupdate.end(), vec_comm_allocator);

  // we create pointers to underlying data
  auto _clusterWeight = usm_clusterWeight.data();
  auto _localCupdate = usm_localCupdate.data();

  q.submit([&](sycl::handler &h){
    h.parallel_for(nv, [=](sycl::id<1> i){
      _clusterWeight[i] = 0;
      _localCupdate[i].degree = 0;
      _localCupdate[i].size = 0;
    });
  }).wait();

  std::memcpy(clusterWeight.data(), usm_clusterWeight.data(), usm_clusterWeight.size() * sizeof(GraphWeight));
  std::memcpy(localCupdate.data(), usm_localCupdate.data(), usm_localCupdate.size() * sizeof(Comm));


} // distCleanCWandCU

#if defined(USE_MPI_RMA)
void fillRemoteCommunities(const Graph &dg, const int me, const int nprocs,
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, const std::vector<GraphElem> &currComm, 
        const std::vector<Comm> &localCinfo, std::map<GraphElem,Comm> &remoteCinfo, 
        std::unordered_map<GraphElem, GraphElem> &remoteComm, std::map<GraphElem,Comm> &remoteCupdate, 
        const MPI_Win &commwin, const std::vector<GraphElem> &disp)
#else
void fillRemoteCommunities(const Graph &dg, const int me, const int nprocs,
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, const std::vector<GraphElem> &currComm, 
        const std::vector<Comm> &localCinfo, std::map<GraphElem,Comm> &remoteCinfo, 
        std::unordered_map<GraphElem, GraphElem> &remoteComm, std::map<GraphElem,Comm> &remoteCupdate)
#endif
{
#if defined(USE_MPI_RMA)
    std::vector<GraphElem> scdata(ssz);
#else
    std::vector<GraphElem> rcdata(rsz), scdata(ssz);
#endif
  GraphElem spos, rpos;
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector< std::vector< GraphElem > > rcinfo(nprocs);
#else
  std::vector<std::unordered_set<GraphElem> > rcinfo(nprocs);
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
  std::vector<GraphElem, vec_ge_alloc> usm_svdata(svdata.begin(), svdata.end(), vec_ge_allocator);
  std::vector<GraphElem, vec_ge_alloc> usm_scdata(scdata.begin(), scdata.end(), vec_ge_allocator);
  std::vector<GraphElem, vec_ge_alloc> usm_currComm(currComm.begin(), currComm.end(), vec_ge_allocator);
  auto _svdata = usm_svdata.data();
  auto _scdata = usm_scdata.data();
  auto _currComm = usm_currComm.data();

  q.submit([&](sycl::handler &h){
    h.parallel_for(ssz, [=](sycl::id<1> i){
      const GraphElem vertex = _svdata[i];
#ifdef DEBUG_PRINTF
      assert((vertex >= base) && (vertex < bound));
#endif
      const GraphElem comm = _currComm[vertex - base];
      _scdata[i] = comm;
    });

  }).wait();

  std::memcpy(scdata.data(), usm_scdata.data(), usm_scdata.size() * sizeof(GraphElem));
  // End SYCL Port

  std::vector<GraphElem> rcsizes(nprocs), scsizes(nprocs);
  std::vector<CommInfo> sinfo, rinfo;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
#if !defined(USE_MPI_RMA) || defined(USE_MPI_ACCUMULATE)
  spos = 0;
  rpos = 0;
#endif
#if defined(USE_MPI_COLLECTIVES)
  std::vector<int> scnts(nprocs), rcnts(nprocs), sdispls(nprocs), rdispls(nprocs);
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

  remoteComm.clear();
  for (GraphElem i = 0; i < rpos; i++) {

#if defined(USE_MPI_RMA)
    const GraphElem comm = rcbuf[i];
#else
    const GraphElem comm = rcdata[i];
#endif

    remoteComm.insert(std::unordered_map<GraphElem, GraphElem>::value_type(rvdata[i], comm));
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
  int local_group_size = 4;

  // Porting to SYCL
  GraphElem *_stcsz = sycl::malloc_host<GraphElem>(1, q);
  *_stcsz = stcsz;
  std::vector<GraphElem, vec_ge_alloc> usm_scsizes(scsizes.begin(), scsizes.end(), vec_ge_allocator);

#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector< std::vector< GraphElem >, vec_vec_ge_alloc > usm_rcinfo(rcinfo.begin(), rcinfo.end(), vec_vec_ge_allocator);
#else
  std::vector<std::unordered_set<GraphElem>, vec_uset_ge_alloc > usm_rcinfo(rcinfo.begin(), rcinfo.end(), vec_uset_ge_allocator);
#endif

  auto _scsizes = usm_scsizes.data();
  auto _rcinfo = usm_rcinfo.data();

  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{nprocs, local_group_size},
      sycl::reduction(_stcsz, std::plus<>()),
      [=](sycl::nd_item<1> it, auto &_stcsz){
      int i = it.get_global_id(0);
      _scsizes[i] = _rcinfo[i].size();
      _stcsz += _scsizes[i];
    });
  }).wait();

  stcsz = *_stcsz;
  std::memcpy(scsizes.data(), usm_scsizes.data(), usm_scsizes.size() * sizeof(GraphElem));
  sycl::free(_stcsz, q);
  // End SYCL port

  MPI_Alltoall(scsizes.data(), 1, MPI_GRAPH_TYPE, rcsizes.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif


  // SYCL Port
  std::vector<GraphElem, vec_ge_alloc> usm_rcsizes(rcsizes.begin(), rcsizes.end(), vec_ge_allocator);
  auto _rcsizes = usm_rcsizes.data();

  GraphElem *_rtcsz = sycl::malloc_host<GraphElem>(1, q);
  *_rtcsz = rtcsz;

  // TODO: Replace explicit group size with default SYCL runtime group size selection
  q.submit([&](sycl::handler &h){
    h.parallel_for(
      sycl::nd_range<1>{nprocs, local_group_size},
      sycl::reduction(_rtcsz, std::plus<>()),
      [=](sycl::nd_item<1> it, auto& _rtcsz) {
        int i = it.get_global_id(0);
        _rtcsz += _rcsizes[i];
    });
  }).wait();

  rtcsz = *_rtcsz;
  // SYCL Port End
  // std::cout << "End of SYCL Port" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  

#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]Total communities to receive: " << rtcsz << std::endl;
#endif
#if defined(USE_MPI_COLLECTIVES)
  std::vector<GraphElem> rcomms(rtcsz), scomms(stcsz);
#else
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector<GraphElem> rcomms(rtcsz);
#else
  std::vector<GraphElem> rcomms(rtcsz), scomms(stcsz);
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
        std::vector<GraphElem, vec_ge_alloc> usm_rcomms(rcomms.begin(), rcomms.end(), vec_ge_allocator);
        std::vector<Comm, vec_comm_alloc> usm_localCinfo(localCinfo.begin(), localCinfo.end(), vec_comm_allocator);
        std::vector<CommInfo, vec_commi_alloc> usm_sinfo(sinfo.begin(), sinfo.end(), vec_commi_allocator);
        std::vector<int, vec_int_alloc> usm_rdispls(rdispls.begin(), rdispls.end(), vec_int_allocator);
        auto _rcomms = usm_rcomms.data();
        auto _localCinfo = usm_localCinfo.data();
        auto _sinfo = usm_sinfo.data();
        auto _rdispls = usm_rdispls.data();

        q.submit([&](sycl::handler &h){
          h.parallel_for(rcsizes[i], [=](sycl::id<1> j){
            const GraphElem comm = _rcomms[_rdispls[i] + j];
            _sinfo[_rdispls[i] + j] = {comm, _localCinfo[comm-base].size, _localCinfo[comm-base].degree};
          });
        }).wait();

        std::memcpy(sinfo.data(), usm_sinfo.data(), usm_sinfo.size() * sizeof(CommInfo));
        // End of port
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
        std::vector<GraphElem, vec_ge_alloc> usm_rcomms(rcomms.begin(), rcomms.end(), vec_ge_allocator);
        std::vector<Comm, vec_comm_alloc> usm_localCinfo(localCinfo.begin(), localCinfo.end(), vec_comm_allocator);
        std::vector<CommInfo, vec_commi_alloc> usm_sinfo(sinfo.begin(), sinfo.end(), vec_commi_allocator);
        auto _rcomms = usm_rcomms.data();
        auto _localCinfo = usm_localCinfo.data();
        auto _sinfo = usm_sinfo.data();

        q.submit([&](sycl::handler &h){
          h.parallel_for(rcsizes[i], [=](sycl::id<1> j){
            const GraphElem comm = _rcomms[rpos + j];
            _sinfo[rpos + j] = {comm, _localCinfo[comm-base].size, _localCinfo[comm-base].degree};
          });
        }).wait();

        std::memcpy(sinfo.data(), usm_sinfo.data(), usm_sinfo.size() * sizeof(CommInfo));
        // End of port
          
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
          std::vector<GraphElem, vec_ge_alloc> usm_rcomms(rcomms.begin(), rcomms.end(), vec_ge_allocator);
          std::vector<Comm, vec_comm_alloc> usm_localCinfo(localCinfo.begin(), localCinfo.end(), vec_comm_allocator);
          std::vector<CommInfo, vec_commi_alloc> usm_sinfo(sinfo.begin(), sinfo.end(), vec_commi_allocator);
          auto _rcomms = usm_rcomms.data();
          auto _localCinfo = usm_localCinfo.data();
          auto _sinfo = usm_sinfo.data();

          q.submit([&](sycl::handler &h){
            h.parallel_for(rcsizes[i], [=](sycl::id<1> j){
              const GraphElem comm = _rcomms[rpos + j];
              _sinfo[rpos + j] = {comm, _localCinfo[comm-base].size, _localCinfo[comm-base].degree};
            });
          }).wait();

          std::memcpy(sinfo.data(), usm_sinfo.data(), usm_sinfo.size() * sizeof(CommInfo));
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

  for (GraphElem i = 0; i < stcsz; i++) {
      const GraphElem ccomm = rinfo[i].community;

      Comm comm;

      comm.size = rinfo[i].size;
      comm.degree = rinfo[i].degree;

      remoteCinfo.insert(std::map<GraphElem,Comm>::value_type(ccomm, comm));
      remoteCupdate.insert(std::map<GraphElem,Comm>::value_type(ccomm, Comm()));
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

void updateRemoteCommunities(const Graph &dg, std::vector<Comm> &localCinfo,
			     const std::map<GraphElem,Comm> &remoteCupdate,
			     const int me, const int nprocs)
{
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  std::vector<std::vector<CommInfo>> remoteArray(nprocs);
  MPI_Comm gcomm = dg.get_comm();
  
  // FIXME TODO can we use TBB::concurrent_vector instead,
  // to make this parallel; first we have to get rid of maps
  for (std::map<GraphElem,Comm>::const_iterator iter = remoteCupdate.begin(); iter != remoteCupdate.end(); iter++) {
      const GraphElem i = iter->first;
      const Comm &curr = iter->second;

      const int tproc = dg.get_owner(i);

#ifdef DEBUG_PRINTF  
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


  // Ported to SYCL
  {
    std::vector<GraphElem, vec_ge_alloc> usm_send_sz(send_sz.begin(), send_sz.end(), vec_ge_allocator);
    std::vector<std::vector<CommInfo>, vec_vec_commi_alloc> usm_remoteArray(remoteArray.begin(), remoteArray.end(), vec_vec_commi_allocator);
    auto _send_sz = usm_send_sz.data();
    auto _remoteArray = usm_remoteArray.data();
    q.submit([&](sycl::handler &h){
      h.parallel_for(nprocs, [=](sycl::id<1> i){
        _send_sz[i] = _remoteArray[i].size();
      });
    }).wait();

    std::memcpy(send_sz.data(), usm_send_sz.data(), usm_send_sz.size() *sizeof(GraphElem));
    // End of Port
  }


  MPI_Alltoall(send_sz.data(), 1, MPI_GRAPH_TYPE, recv_sz.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

#ifdef DEBUG_PRINTF  
  const double t1 = MPI_Wtime();
  tc += (t1 - t0);
#endif

  GraphElem rcnt = 0, scnt = 0;
  auto _rcnt = sycl::malloc_host<GraphElem>(1, q);
  auto _scnt = sycl::malloc_host<GraphElem>(1, q);
  
  *_scnt = scnt;
  *_rcnt = rcnt;

  // Porting to SYCL
  {
    int local_group_size = 4;
    std::vector<GraphElem, vec_ge_alloc> usm_recv_sz(recv_sz.begin(), recv_sz.end(), vec_ge_allocator);
    std::vector<GraphElem, vec_ge_alloc> usm_send_sz(send_sz.begin(), send_sz.end(), vec_ge_allocator);
    auto _send_sz = usm_send_sz.data();
    auto _recv_sz = usm_recv_sz.data();

    q.submit([&](sycl::handler &h){
      h.parallel_for(
          sycl::nd_range<1>{nprocs, local_group_size},
          sycl::reduction(_rcnt, std::plus<>()),
          [=](sycl::nd_item<1> it, auto& _rcnt) {
            int i = it.get_global_id(0);
            _rcnt += _recv_sz[i];
      });
    });

    q.submit([&](sycl::handler &h){
      h.parallel_for(
          sycl::nd_range<1>{nprocs, local_group_size},
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


// SYCL port
std::vector<Comm, vec_comm_alloc> usm_localCinfo(localCinfo.begin(), localCinfo.end(), vec_comm_allocator);
std::vector<CommInfo, vec_commi_alloc> usm_rdata(rdata.begin(), rdata.end(), vec_commi_allocator);
auto _localCinfo = usm_localCinfo.data();
auto _rdata = usm_rdata.data();

q.submit([&](sycl::handler &h){
  h.parallel_for(rcnt, [=](sycl::id<1> i){
    const CommInfo &curr = _rdata[i];
#ifdef DEBUG_PRINTF  
    assert(dg.get_owner(curr.community) == me);
#endif
    _localCinfo[curr.community-base].size += curr.size;
    _localCinfo[curr.community-base].degree += curr.degree;
  });
}).wait();

std::memcpy(localCinfo.data(), usm_localCinfo.data(), usm_localCinfo.size() * sizeof(Comm));
// End port

} // updateRemoteCommunities

// initial setup before Louvain iteration begins
#if defined(USE_MPI_RMA)
void exchangeVertexReqs(const Graph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        const int me, const int nprocs, MPI_Win &commwin)
#else
void exchangeVertexReqs(const Graph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        const int me, const int nprocs)
#endif
{
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  std::vector<std::unordered_set<GraphElem>> parray(nprocs);

  // TODO: Did not port the below into SYCL
  // If there is time, we'll attempt to play around with a parallelized version
  // It seems that for CPU, the below is faster than any workaround we come up with
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
  // TODO FIXME parallelize this loop
  for (std::vector<std::unordered_set<GraphElem>>::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
    ssz += iter->size();
    ssizes[pproc] = iter->size();
    pproc++;
  }

  MPI_Alltoall(ssizes.data(), 1, MPI_GRAPH_TYPE, rsizes.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

  GraphElem rsz_r = 0;
  
  // SYCL Port
  std::vector<GraphElem, vec_ge_alloc> usm_rsizes(rsizes.begin(), rsizes.end(), vec_ge_allocator);
  GraphElem *_rsz_r = sycl::malloc_host<GraphElem>(1, q);
  *_rsz_r = rsz_r;
  auto _rsizes = usm_rsizes.data();

  int local_group_size = 4;
  q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::nd_range<1> {nprocs, local_group_size},
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
  std::swap(ssizes, rsizes);
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
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata, const GraphWeight lower, 
        const GraphWeight thresh, int &iters, MPI_Win &commwin)
#else
GraphWeight distLouvainMethod(const int me, const int nprocs, const Graph &dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata, const GraphWeight lower, 
        const GraphWeight thresh, int &iters)
#endif
{
  std::vector<GraphElem> pastComm, currComm, targetComm;
  std::vector<GraphWeight> vDegree;
  std::vector<GraphWeight> clusterWeight;
  std::vector<Comm> localCinfo, localCupdate;
 
  std::unordered_map<GraphElem, GraphElem> remoteComm;
  std::map<GraphElem,Comm> remoteCinfo, remoteCupdate;
  
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;
  
  // Ported to SYCL
  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
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
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs, commwin);
  
  // store the remote displacements 
  std::vector<GraphElem> disp(nprocs);
  MPI_Exscan(ssizes.data(), (GraphElem*)disp.data(), nprocs, MPI_GRAPH_TYPE, 
          MPI_SUM, gcomm);
#else
  // TODO: Port to SYCL
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
    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate, 
            commwin, disp);
#else
    // TODO: Port to SYCL
    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate);
#endif

    //std::cout "Executed fillRemoteCommunities(...) " << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    std::cout << "[" << me << "]Remote community map size: " << remoteComm.size() << std::endl;
    std::cout << "[" << me << "]Iteration communication time: " << (t1 - t0) << std::endl;
#endif

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

    // Ported to SYCL
    distCleanCWandCU(nv, clusterWeight, localCupdate);
    //std::cout "Cleaned CW and CU" << std::endl;

// NOTE: The distExecuteLouvain Iteration cannot be ported until we complete the following
// TODO: Get experience with atomics and locking and porting these from OpenMP to SYCL
#pragma omp parallel default(shared), shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate), \
        firstprivate(constantForSecondTerm, me)
    {

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(guided) 
#endif
        for (GraphElem i = 0; i < nv; i++) {
            distExecuteLouvainIteration(i, dg, currComm, targetComm, vDegree, localCinfo, 
                    localCupdate, remoteComm, remoteCinfo, remoteCupdate,
                    constantForSecondTerm, clusterWeight, me);
        }
    }


    // Ported to SYCL
    //std::cout "distUpdateLocalCinfo" << std::endl;
    distUpdateLocalCinfo(localCinfo, localCupdate);

    // communicate remote communities (TODO: Need to port)
    //std::cout "updateRemoteCommunities" << std::endl;
    updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);

    // compute modularity (TODO: Need to port)
    //std::cout "Compute modularity" << std::endl;
    currMod = distComputeModularity(dg, localCinfo, clusterWeight, constantForSecondTerm, me);

    // exit criteria
    if (currMod - prevMod < thresh)
        break;

    prevMod = currMod;
    if (prevMod < lower)
        prevMod = lower;

    // Define sycl USM constructs
    //std::cout "Louvain method iteration cleanup" << std::endl;
    std::vector<GraphElem, vec_ge_alloc> usm_pastComm(pastComm.begin(), pastComm.end(), vec_ge_allocator);
    std::vector<GraphElem, vec_ge_alloc> usm_currComm(currComm.begin(), currComm.end(), vec_ge_allocator);
    std::vector<GraphElem, vec_ge_alloc> usm_targetComm(targetComm.begin(), targetComm.end(), vec_ge_allocator);
    auto _pastComm = usm_pastComm.data();
    auto _currComm = usm_currComm.data();
    auto _targetComm = usm_targetComm.data();

    q.submit([&](sycl::handler &h){
      h.parallel_for(nv, [=](sycl::id<1> i){
        GraphElem tmp = _pastComm[i];
        _pastComm[i] = _currComm[i];
        _currComm[i] = _targetComm[i];
        _targetComm[i] = tmp;
      });
    }).wait();
    
    // Update original STL containers
    std::memcpy(pastComm.data(), usm_pastComm.data(), usm_pastComm.size() * sizeof(GraphElem));
    std::memcpy(currComm.data(), usm_currComm.data(), usm_currComm.size() * sizeof(GraphElem));
    std::memcpy(targetComm.data(), usm_targetComm.data(), usm_targetComm.size() * sizeof(GraphElem));

  } // end of Louvain iteration

  //std::cout "Louvain method exit for loop" << std::endl;

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
  
  return prevMod;
  //std::cout "Louvain method exit function" << std::endl;

} // distLouvainMethod plain

#endif // __DSPL
