[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://doi.org/10.1287/ijoc.2023.1288)

# Decomposition Strategies for Vehicle Routing Heuristics

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data used in the paper [Decomposition Strategies for Vehicle Routing Heuristics](https://doi.org/10.1287/ijoc.2023.1288) by A. Santini, M. Schneider, T. Vidal, and D. Vigo.
The snapshot is based on [alberto-santini/cvrp-decomposition v0.1](https://github.com/alberto-santini/cvrp-decomposition/releases/tag/0.1) in the development repository. 
Users are advised to use [cvrp-decomposition](https://github.com/alberto-santini/cvrp-decomposition), which might contain improvements and fixes compared to this repository.

This repository presents a proof-of-concept for including instance decomposition within two popular metaheuristics for the Capacitated Vehicle Routing Problem (CVRP):
* The Adaptive Large Neighbourhood Search of Stefan Ropke
* The Hybrid Genetic Algorithm of Thibaut Vidal

We propose several decomposition techniques, which we describe in detail in our paper:

### Cite
```bib
@article{cvrp_decomposition,
    title={Decomposition strategies for vehicle routing heuristics},
    author={Santini, Alberto and Schneider, Michael and Vidal, Thibaut and Vigo, Daniele},
    year=2022,
    journal={{INFORMS Journal on Computing}},
    pubstate={to appear}
}
```

You can cite this repository via GitHub:

### Cite
DOI: [10.1287/ijoc.2023.1288.cd](https://doi.org/10.1287/ijoc.2023.1288.cd)
```bib
@article{cvrp_decomposition_2022,
    title={{Repository cvrp-decomposition} Version v2022.0048},
    author={Ropke, Stefan and Vidal, Thibaut and Santini, Alberto},
    year=2022,
    doi={10.1287/ijoc.2023.1288.cd},
    url={https://github.com/INFORMSJoC/2022.0048}
}
```

## Authors

The original ALNS code is by Stefan Ropke.
Alberto Santini then edited Stefan's code and implemented instance decomposition.
The original paper in which Stefan introduced ALNS is the following:

```bib
@article{ropke_2006,
    title={An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows},
    author={Ropke, Stefan and Pisinger, David},
    year=2006,
    journal={Transportation Science},
    volume=40,
    number=4,
    pages={455--472},
    doi={10.1287/trsc.1050.0135}
}
```

The original HGS code is by Thibaut Vidal.
Thibaut also released his code in repository [vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP) and through the below paper.
Alberto Santini then edited Thibaut's code and implemented instance decomposition.

```bib
@article{vidal_2022,
    title={Hybrid genetic search for the CVRP: Open-source implementation and SWAP\textsupersctipt{*} neighborhood},
    author={Vidal, Thibaut},
    year=2022,
    journal={Computers \& Operations Research},
    volume=140,
    doi={10.1016/j.cor.2021.105643}
}
```

## Usage

### HGS

The preferred way to build the HGS programme is through cmake.
We provide a `CMakeList.txt` file, which allows to build the executable as follows:

```
$ git clone https://github.com:alberto-santini/cvrp-decomposition.git
$ cd cvrp-decomposition
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ cmake
$ make
```

### ALNS

The ALNS solver is distributed in binary form in folder `bin`.
The executable (for Linux x86-64) is `palns-cvrp`.
Running the executable file without any parameter displays the available parameters:

```
-f <arg>    : Load problem given by <arg>
-t <arg>    : problem type to load: 
              1 = JFC CVRP (default)
              2 = GWKC
              3 = TSPLIB format (used by exact algs.)
              4 = Tailard
              5 = TSPLIB format, but #vehicles is free (for Uchoa et al. instances)
-p <arg>    : Use parameter file given by <arg>
-P <arg>    : override parameters from par file with <arg>
-n <arg>    : number of threads to use.
-v <arg>	: Number of vehicles (for type 3 instances)
-r <arg>	: Number of retries
-h          : This help
-H          : Add header line to output file
-o <arg>    : prefix of summary files
```

## Instances

Instances contained in folder `data` are part of the so-called "Uchoa dataset".
We refer to the following paper for further information on their generation and characteristics:

```bib
@article{uchoa_2017,
    title={New benchmark instances for the capacitated vehicle routing problem},
    author={Uchoa, Eduardo and Pecin, Diego and Pessoa, Artur and Poggi, Marcus and Vidal, Thibaut and Subramanian, Anand},
    year=2017,
    journal={European Journal of Operational Research},
    volume=257,
    number=3,
    pages={845--858},
    doi={10.1016/j.ejor.2016.08.012}
}
```

## Results

Folder `results` contains the figures used in the paper.

## Acknowledgements

We are extremely grateful to Stefan Ropke for sharing with us his code for ALNS applied to the CVRP.

## License

This software is released under the MIT license, which we report in file `LICENSE`.
