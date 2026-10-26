# Task catalogue

Generated from source by `scripts/build_task_catalogue.py`.

There are **66 task interfaces**, including **36 CO-Bench interfaces** and **120 registered Feynman equation definitions** behind the Feynman interface. These are different counting units.

The earlier “100+ tasks” description includes wrapped external suites; it must not be read as 100+ independently implemented native task interfaces. Dataset instances, equation IDs and parameter variations are not added to the interface count.

This inventory does not certify semantic correctness or dataset licensing. Rows pending manual review must not be represented as validated benchmark definitions. AILS2 is not identified by that name in the current source tree; any external reference to it must be mapped to a concrete implementation.

| Interface | Class | Suite | Semantic review |
| --- | --- | --- | --- |
| `machine_learning/acrobot` | `AcrobotEvaluation` | Other | pending manual review |
| `machine_learning/car_mountain` | `CarMountainEvaluation` | Other | pending manual review |
| `machine_learning/car_mountain_continue` | `CarMountainCEvaluation` | Other | pending manual review |
| `machine_learning/car_racing` | `RacingCarEvaluation` | Other | pending manual review |
| `machine_learning/moon_lander` | `MoonLanderEvaluation` | Other | pending manual review |
| `machine_learning/pendulum` | `PendulumEvaluation` | Other | pending manual review |
| `optimization/admissible_set` | `ASPEvaluation` | Other | checked against evaluator |
| `optimization/bp_1d_construct` | `BP1DEvaluation` | Other | pending manual review |
| `optimization/bp_2d_construct` | `BP2DEvaluation` | Other | checked against evaluator |
| `optimization/cflp_construct` | `CFLPEvaluation` | Other | checked against evaluator |
| `optimization/co_bench/aircraft_landing_co_bench` | `ALEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/assignment_problem_co_bench` | `APEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/assortment_problem_co_bench` | `AssortPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/bp_1d_co_bench` | `BP1DEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/capacitated_warehouse_location_co_bench` | `CWLEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/common_due_date_scheduling_co_bench` | `CDDSEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/constrained_guillotine_cutting_co_bench` | `CGCEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/constrained_non_guillotine_cutting_co_bench` | `CNCEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/container_loading_co_bench` | `CLEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/container_loading_with_weight_restrictions_co_bench` | `CLWREvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/corporate_structuring_co_bench` | `CSEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/crew_scheduling_co_bench` | `CSchedulingEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/equitable_partitioning_problem_co_bench` | `EPPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/euclidean_steiner_problem_co_bench` | `ESPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/flow_shop_scheduling_co_bench` | `FSSEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/generalised_assignment_problem_co_bench` | `GAPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/graph_colouring_co_bench` | `GCEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/hybrid_reentrant_shop_scheduling_co_bench` | `HRSSEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/job_shop_scheduling_co_bench` | `JSSEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/maximal_independent_set_co_bench` | `MISEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/multi_demand_multidimensional_knapsack_problem_co_bench` | `MDMKPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/multidimensional_knapsack_problem_co_bench` | `MKPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/open_shop_scheduling_co_bench` | `OSSEvaluationCB` | CO-Bench | checked against evaluator |
| `optimization/co_bench/p_median_capacitated_co_bench` | `PMCEvaluationCB` | CO-Bench | checked against evaluator |
| `optimization/co_bench/p_median_uncapacitated_co_bench` | `PMUEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/packing_unequal_circles_area_co_bench` | `PUCAEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/packing_unequal_circles_co_bench` | `PUCEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/packing_unequal_rectangles_and_squares_area_co_bench` | `PURSAEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/packing_unequal_rectangles_and_squares_co_bench` | `PURSEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/resource_constrained_shortest_path_co_bench` | `RCSPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/set_covering_co_bench` | `SCEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/set_partitioning_co_bench` | `SPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/travelling_salesman_problem_co_bench` | `TSPEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/uncapacitated_warehouse_location_co_bench` | `UWLEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/unconstrained_guillotine_cutting_co_bench` | `UGCEvaluationCB` | CO-Bench | pending manual review |
| `optimization/co_bench/vehicle_routing_period_routing_co_bench` | `VRPREvaluationCB` | CO-Bench | pending manual review |
| `optimization/cvrp_construct` | `CVRPEvaluation` | Other | pending manual review |
| `optimization/jssp_construct` | `JSSPEvaluation` | Other | pending manual review |
| `optimization/knapsack_construct` | `KnapsackEvaluation` | Other | pending manual review |
| `optimization/max_cut` | `MaxCutEvaluation` | Other | pending manual review |
| `optimization/online_bin_packing` | `OBPEvaluation` | Other | pending manual review |
| `optimization/online_bin_packing_2O` | `OBP_2O_Evaluation` | Other | checked against evaluator |
| `optimization/orienteering_construct` | `OrienteeringEvaluation` | Other | pending manual review |
| `optimization/ovrp_construct` | `OVRPEvaluation` | Other | pending manual review |
| `optimization/pymoo_moead` | `MOEAD_PYMOO_Evaluation` | Other | checked against evaluator |
| `optimization/qap_construct` | `QAPEvaluation` | Other | pending manual review |
| `optimization/set_cover_construct` | `SCPEvaluation` | Other | checked against evaluator |
| `optimization/tsp_construct` | `TSPEvaluation` | Other | pending manual review |
| `optimization/tsp_gls_2O` | `TSP_GLS_2O_Evaluation` | Other | pending manual review |
| `optimization/vrptw_construct` | `VRPTWEvaluation` | Other | pending manual review |
| `science_discovery/bactgrow` | `BGEvaluation` | Other | pending manual review |
| `science_discovery/feynman_srsd` | `FeynmanEvaluation` | Feynman SRSD | pending manual review |
| `science_discovery/ode_1d` | `ODEEvaluation` | Other | pending manual review |
| `science_discovery/oscillator1` | `OscillatorEvaluation1` | Other | pending manual review |
| `science_discovery/oscillator2` | `OscillatorEvaluation2` | Other | pending manual review |
| `science_discovery/stresstrain` | `SSEvaluation` | Other | pending manual review |

## Checked definitions

- **Priority search for symmetric constant-weight admissible sets**: Score candidate vectors to construct a pre-admissible set, then expand the symmetric groups. Maximize the expanded set cardinality minus the configured reference cardinality; see TRIPLES and surviving-child checks in the evaluator.
- **Constructive rectangular 2D bin packing**: Place every item without overlap inside finite bins; maximize negative mean used-bin count. Incomplete packing returns None.
- **Constructive capacitated facility assignment**: Assign every customer; maximize negative mean assignment cost. Incomplete assignment returns None. The current objective sums assignment costs, not a separate facility-opening cost.
- **CO-Bench open shop scheduling**: Respect job/machine non-overlap; maximize mean lower-bound/makespan ratio.
- **CO-Bench capacitated p-median**: Select p customer medians and feasible capacity-limited assignments; maximize mean best-known/computed-cost ratio.
- **Two-objective online bin packing**: Minimize mean bin count and mean evaluation wall-clock time; both returned with negative signs. 2O means two objectives, not two-orthant constraints. Runtime is hardware-dependent.
- **Pymoo MOEA/D decomposition design**: Design a decomposition function for MOEA/D on generated DTLZ4 instances; maximize hypervolume.
- **Constructive set cover**: Cover every universal-set element using supplied subsets; maximize negative mean number of selected subsets.
