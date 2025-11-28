cmake_minimum_required(VERSION 3.15.0)
project(decomposition VERSION 0.0.1 LANGUAGES CXX)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR})

set(WARNING_OPTIONS             -Wall -Wextra -Werror)
set(RELEASE_OPTIONS             -O3 -march=native -mtune=native)
set(DEBUG_OPTIONS               -O0 -g)
set(GDB_DEBUG_OPTIONS           -ggdb)
set(LINKER_OPTIONS              -flto)

add_compile_options(${WARNING_OPTIONS})
add_compile_options("$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")
add_compile_options("$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
add_compile_options("$<$<AND:$<CONFIG:DEBUG>,$<CXX_COMPILER_ID:GNU>>:${GDB_DEBUG_OPTIONS}>")
add_compile_definitions("$<$<AND:$<CONFIG:DEBUG>,$<CXX_COMPILER_ID:GNU>>:${GCC_DEBUG_DEFINES}>")

# add_library(srp
#         src/alns/SRP-Utils/ArgParser.h
#         src/alns/SRP-Utils/Coordinate.h
#         src/alns/SRP-Utils/GraphObjects.h
#         src/alns/SRP-Utils/HashNumberGenerator.h
#         src/alns/SRP-Utils/LimIntSet.h
#         src/alns/SRP-Utils/Matrix.h
#         src/alns/SRP-Utils/NSMatrix.h
#         src/alns/SRP-Utils/ParFileParser.h
#         src/alns/SRP-Utils/StatUtils.h
#         src/alns/SRP-Utils/Utils.h
#         src/alns/SRP-Utils/VectorUtils.h
#         src/alns/SRP-Utils/ArgParser.cpp
#         src/alns/SRP-Utils/GraphObjects.cpp
#         src/alns/SRP-Utils/HashNumberGenerator.cpp
#         src/alns/SRP-Utils/ParFileParser.cpp
#         src/alns/SRP-Utils/Utils.cpp
#         src/alns/SRP-Utils/VectorUtils.cpp)

# target_compile_features(    srp PUBLIC cxx_std_17)
# target_link_libraries(      srp PUBLIC ${LINKER_OPTIONS})
# target_link_libraries(      srp PUBLIC dl)
# target_compile_options(     srp PRIVATE
#         -Wno-misleading-indentation
#         -Wno-sign-compare
#         -Wno-unused-parameter
#         -Wno-unused-variable
#         -Wno-parentheses
#         -Wno-reorder
#         -Wno-unused-but-set-variable
#         -Wno-maybe-uninitialized
#         -Wno-deprecated-copy)

# add_library(palns
#         src/alns/PALNS/PALNS.h
#         src/alns/PALNS/PALNS.cpp
#         src/alns/PALNS/PALNSAcceptanceCriterion.h
#         src/alns/PALNS/PALNSHelpers.h
#         src/alns/PALNS/PALNSParameters.h
#         src/alns/PALNS/PALNSParameters.cpp
#         src/alns/PALNS/AcceptanceCriteria/RecordDeviation.h
#         src/alns/PALNS/AcceptanceCriteria/RecordDeviation.cpp
#         src/alns/PALNS/PALNSDecompositionMethod.h
#         src/alns/PALNS/DecompositionMethods/RouteSequenceDecomposition.h
#         src/alns/PALNS/DecompositionMethods/RouteSequence/RandomRouteDecomposition.h
#         src/alns/PALNS/SubProblem.h
#         src/alns/PALNS/SubProblem.cpp
#         src/alns/PALNS/PALNSDecompositionMethod.cpp
#         src/alns/PALNS/DecompositionMethods/RouteSequenceDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/RouteSequence/RandomRouteDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/RouteSequence/BarycentreSwipeDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/RouteSequence/BarycentreSwipeDecomposition.h
#         src/alns/PALNS/DecompositionMethods/RouteSequence/BarycentreQuadrantDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/RouteSequence/BarycentreQuadrantDecomposition.h
#         src/alns/PALNS/DecompositionMethods/RouteSequence/BarycentreClusteringDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/RouteSequence/BarycentreClusteringDecomposition.h
#         src/alns/PALNS/DecompositionMethods/RouteSequence/RouteHistoryDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/RouteSequence/RouteHistoryDecomposition.h
#         src/alns/PALNS/DecompositionMethods/ArcBasedDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/ArcBasedDecomposition.h
#         src/alns/PALNS/DecompositionMethods/ArcBased/RandomArcDecomposition.h
#         src/alns/PALNS/DecompositionMethods/ArcBased/RandomArcDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/ArcBased/RandomPathDecomposition.h
#         src/alns/PALNS/DecompositionMethods/ArcBased/RandomPathDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/ArcBased/CostArcDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/ArcBased/CostArcDecomposition.h
#         src/alns/PALNS/DecompositionMethods/ArcBased/CostPathDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/ArcBased/CostPathDecomposition.h
#         src/alns/PALNS/DecompositionMethods/ArcBased/ArcHistoryDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/ArcBased/ArcHistoryDecomposition.h
#         src/alns/PALNS/DecompositionMethods/ArcBased/PathHistoryDecomposition.cpp
#         src/alns/PALNS/DecompositionMethods/ArcBased/PathHistoryDecomposition.h)

# target_compile_features(    palns PUBLIC cxx_std_17)
# target_link_libraries(      palns PUBLIC ${LINKER_OPTIONS})
# target_link_libraries(      palns PUBLIC srp)
# target_include_directories( palns PUBLIC src/alns/SRP-Utils)
# target_compile_options(     palns PRIVATE
#         -Wno-misleading-indentation
#         -Wno-reorder
#         -Wno-sign-compare
#         -Wno-unused-parameter)

# add_executable(palns-cvrp
#         src/alns/PALNS-CVRP/ArcCostTracker.h
#         src/alns/PALNS-CVRP/CVRPInstance.h
#         src/alns/PALNS-CVRP/CVRPSolution.h
#         src/alns/PALNS-CVRP/CVRPUtils.h
#         src/alns/PALNS-CVRP/DestroyMethods.h
#         src/alns/PALNS-CVRP/GreedySeqAlg.h
#         src/alns/PALNS-CVRP/IncrementalTSP.h
#         src/alns/PALNS-CVRP/InsertionCache.h
#         src/alns/PALNS-CVRP/PALNS-CVRP.h
#         src/alns/PALNS-CVRP/PetalAlg.h
#         src/alns/PALNS-CVRP/Regret2Alg.h
#         src/alns/PALNS-CVRP/RepairMethods.h
#         src/alns/PALNS-CVRP/Route.h
#         src/alns/PALNS-CVRP/SeedAlg.h
#         src/alns/PALNS-CVRP/SeqInsertionCache.h
#         src/alns/PALNS-CVRP/SeqRegret2Alg.h
#         src/alns/PALNS-CVRP/ArcCostTracker.cpp
#         src/alns/PALNS-CVRP/CVRPInstance.cpp
#         src/alns/PALNS-CVRP/CVRPSolution.cpp
#         src/alns/PALNS-CVRP/DestroyMethods.cpp
#         src/alns/PALNS-CVRP/GreedySeqAlg.cpp
#         src/alns/PALNS-CVRP/InsertionCache.cpp
#         src/alns/PALNS-CVRP/main.cpp
#         src/alns/PALNS-CVRP/PALNS-CVRP.cpp
#         src/alns/PALNS-CVRP/Regret2Alg.cpp
#         src/alns/PALNS-CVRP/RepairMethods.cpp
#         src/alns/PALNS-CVRP/Route.cpp
#         src/alns/PALNS-CVRP/SeedAlg.cpp
#         src/alns/PALNS-CVRP/SeqInsertionCache.cpp
#         src/alns/PALNS-CVRP/SeqRegret2Alg.cpp)

# target_compile_features(    palns-cvrp PRIVATE cxx_std_17)
# target_link_libraries(      palns-cvrp PRIVATE ${LINKER_OPTIONS})
# target_link_libraries(      palns-cvrp PRIVATE palns)
# target_include_directories( palns-cvrp PRIVATE src/alns/PALNS)
# target_compile_options(     palns-cvrp PRIVATE
#         -Wno-misleading-indentation
#         -Wno-reorder
#         -Wno-unused-parameter
#         -Wno-sign-compare
#         -Wno-unused-variable
#         -Wno-maybe-uninitialized
#         -Wno-comment
#         -Wno-unused-but-set-variable
#         -Wno-delete-non-virtual-dtor
#         -Wno-deprecated-copy)

add_executable(hgs
        src/hgs-TV/CommandLineParser.h
        src/hgs-TV/DecompositionMethod.h
        src/hgs-TV/Genetic.h
        src/hgs-TV/Params.h
        src/hgs-TV/Individual.h
        src/hgs-TV/LocalSearch.h
        src/hgs-TV/Params.h
        src/hgs-TV/Population.h
        src/hgs-TV/SolverStatus.h
        src/hgs-TV/Split.h
        src/hgs-TV/SubProblem.h
        src/hgs-TV/UchoaVehicles.h
        src/hgs-TV/Decomposition/ArcBasedDecomposition.h
        src/hgs-TV/Decomposition/ArcBased/ArcHistoryDecomposition.h
        src/hgs-TV/Decomposition/RouteSequence/BarycentreClusteringDecomposition.h
        src/hgs-TV/Decomposition/RouteSequence/BarycentreQuadrantDecomposition.h
        src/hgs-TV/Decomposition/RouteSequence/BarycentreSwipeDecomposition.h
        src/hgs-TV/Decomposition/ArcBased/CostArcDecomposition.h
        src/hgs-TV/Decomposition/ArcBased/CostPathDecomposition.h
        src/hgs-TV/Decomposition/ArcBased/RandomPathDecomposition.h
        src/hgs-TV/Decomposition/ArcBased/RandomArcDecomposition.h
        src/hgs-TV/Decomposition/RouteSequence/RandomRouteDecomposition.h
        src/hgs-TV/Decomposition/RouteSequence/RouteHistoryDecomposition.h
        src/hgs-TV/Decomposition/RouteSequenceDecomposition.h
        src/hgs-TV/Decomposition/ClusteringUtils/Clustering.h
        src/hgs-TV/Decomposition/ClusteringUtils/KMeans.h
        src/hgs-TV/Decomposition/ClusteringUtils/KMedoids.h
        src/hgs-TV/CommandLineParser.cpp
        src/hgs-TV/DecompositionMethod.cpp
        src/hgs-TV/Genetic.cpp
        src/hgs-TV/Individual.cpp
        src/hgs-TV/LocalSearch.cpp
        src/hgs-TV/main.cpp
        src/hgs-TV/Params.cpp
        src/hgs-TV/Population.cpp
        src/hgs-TV/Split.cpp
        src/hgs-TV/SubProblem.cpp
        src/hgs-TV/Decomposition/ArcBasedDecomposition.cpp
        src/hgs-TV/Decomposition/ArcBased/ArcHistoryDecomposition.cpp
        src/hgs-TV/Decomposition/RouteSequence/BarycentreClusteringDecomposition.cpp
        src/hgs-TV/Decomposition/RouteSequence/BarycentreQuadrantDecomposition.cpp
        src/hgs-TV/Decomposition/RouteSequence/BarycentreSwipeDecomposition.cpp
        src/hgs-TV/Decomposition/ArcBased/CostArcDecomposition.cpp
        src/hgs-TV/Decomposition/ArcBased/CostPathDecomposition.cpp
        src/hgs-TV/Decomposition/ArcBased/RandomPathDecomposition.cpp
        src/hgs-TV/Decomposition/ArcBased/RandomArcDecomposition.cpp
        src/hgs-TV/Decomposition/RouteSequence/RandomRouteDecomposition.cpp
        src/hgs-TV/Decomposition/RouteSequence/RouteHistoryDecomposition.cpp
        src/hgs-TV/Decomposition/RouteSequenceDecomposition.cpp
        src/hgs-TV/Decomposition/ArcBased/PathHistoryDecomposition.cpp
        src/hgs-TV/Decomposition/ArcBased/PathHistoryDecomposition.h)

target_compile_features(    hgs PRIVATE cxx_std_17)
target_link_libraries(      hgs PRIVATE pthread)
target_link_libraries(      hgs PRIVATE dl)
target_compile_options(     hgs PRIVATE
        -Wno-sign-compare)

option(BATCH_MODE "Suppress printing to stdout, for running experiments on clusters." OFF)
if(BATCH_MODE)
    target_compile_definitions(hgs PRIVATE BATCH_MODE)
#     target_compile_definitions(palns PRIVATE BATCH_MODE)
#     target_compile_definitions(palns-cvrp PRIVATE BATCH_MODE)
endif()