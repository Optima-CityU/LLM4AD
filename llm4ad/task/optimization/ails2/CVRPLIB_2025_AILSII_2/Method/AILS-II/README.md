![](https://camo.githubusercontent.com/1b8f04b8ff248ffd132c13343858d070c4805406bbd4c4651f9b27e9c2f01a58/68747470733a2f2f494e464f524d534a6f432e6769746875622e696f2f6c6f676f732f494e464f524d535f4a6f75726e616c5f6f6e5f436f6d707574696e675f4865616465722e6a7067) 
# AILS-II: An Adaptive Iterated Local Search Heuristic for the Large-scale Capacitated Vehicle Routing Problem

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper AILS-II: An Adaptive Iterated Local Search Heuristic for the Large-Scale Capacitated Vehicle Routing Problem by V. R. Máximo, J. F. Cordeau and M. C. V. Nascimento. 

Important: This code is being developed on an on-going basis at [https://github.com/vinymax10/AILS-CVRP](https://github.com/vinymax10/AILS-CVRP). Please go there if you would like to get a more recent version or would like support

## Cite

To cite the contents of this repository, please cite this the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0106

https://doi.org/10.1287/ijoc.2023.0106.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{Maximoetal2024,
  author =        {V. R. Máximo and J.-F.Courdeau and M. C. V. Nascimento},
  publisher =     {INFORMS Journal on Computing},
  title =         {AILS-II: An Adaptive Iterated Local Search Heuristic for the Large-scale Capacitated Routing Problem},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0106.cd},
  url =           {https://github.com/INFORMSJoC/2023.0106},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0106},
}  
```
## Other References

Besides citing the paper and this repo, those interested in using any part of this algorithm in academic works must cite the following references:

[1] Máximo, Vinícius R., Nascimento, Mariá C.V. (2021).
A hybrid adaptive iterated local search with diversification control to the capacitated vehicle routing problem. European Journal of Operational Research, Volume 294, p. 1108-1119, https://doi.org/10.1016/j.ejor.2021.02.024 (also available at [aXiv](https://arxiv.org/abs/2012.11021)).

[2] Máximo, Vinícius R., Cordeau, Jean-François, Nascimento, Mariá C.V. (2022).
An adaptive iterated local search heuristic for the Heterogeneous Fleet Vehicle Routing Problem. Computers & Operations Research, Volume 148, p. 105954.
https://doi.org/10.1016/j.cor.2022.105954 (also available at [aXiv](https://arxiv.org/abs/2111.12821)).

## Description

AILS-II is an Adaptive Iterated Local Search (AILS) meta-heuristic that embeds adaptive strategies to tune  diversity control parameters. These parameters are the perturbation degree and the acceptance criterion. They are key parameters to ensure that the method escapes from local optima and keeps an adequate level of exploitation and exploration of the method. Its implementation is in JAVA language.

## To run the algorithm

```console
java -jar AILSII.jar -file data/X-n214-k11.vrp -rounded true -best 10856 -limit 100 -stoppingCriterion Time 
```



# 创建输出目录
mkdir out\production\AILS-II

# 编译代码
javac -d out\production\AILS-II -cp src (Get-ChildItem src -Filter *.java -Recurse).FullName

# 打包 JAR 文件
jar cvfe AILSII.jar SearchMethod.AILSII -C out\production\AILS-II .

# 测
java -jar AILSII.jar -file XLDemo/XLTEST-n1048-k139.vrp -rounded true -best 10856 -limit 100 -stoppingCriterion Time 

Run the AILSII class that has the following parameters:

**-file** : the file address of the problem instance.

**-rounded** :  A flag that indicates whether the instance has rounded distances or not. The options are: [false, true]. The default value is true.

**-stoppingCriterion** : It is possible to use two different stopping criteria:
* **Time** : The algorithm stops when a given time in seconds has elapsed; 
* **Iteration** :  The algorithm stops when the given number of iterations has been reached. 

**-limit** : Refers to the value that will be used in the stopping criterion. If the stopping criterion is a time limit, this parameter is the timeout in seconds. Otherwise, this parameter indicates the number of iterations. The default value is the maximum limit for a double precision number in the JAVA language (Double.MAX_VALUE).

**-best** :  Indicates the value of the optimal solution. The default value is 0.

**-varphi** :  Parameter of the feasibility and local search methods that refers to the maximum cardinality of the set of nearest neighbors of the vertices. The default value is 40. The larger it is, the greater the number of movements under consideration in the methods. 

**-gamma** :  Number of iterations for AILS-II to perform a new adjustment of variable 𝜔. The default value is 30.

**-dMax** : Initial reference distance between the reference solution and the  solution obtained after the local search. The default value is 30.

**-dMin** : Final Reference distance between the reference solution and the solution obtained after the local search. The default value is 15.

## Data

The folder data contains all literature instances tested (files with extension .vrp). The files with the extension .sol refer to the Best Known Solutions (BKSs) used to calculate the gaps.

## Results

The folder results contain all tables and figures elaborated after carrying out the experiments using the instances found in the folder data. To achieve the results presented in [Table 2](results/Table2.png) and [Table 3 (part 1)](results/Table3_1.png), [Table 3 (part 2)](results/Table3_2.png) and [Table 3 (part 3)](results/Table3_3.png), we ran the scripts in folder [scripts](scripts). The script [scriptAilsIIUCHOA.sh](scripts/scriptAilsIIUCHOA.sh) refer to the tests performed with the X instances, whereas the script [scriptAilsIIArnold.sh](scripts/scriptAilsIIArnold.sh) run the remainder instances. 



				System.out.println("solution quality: "+bestF
				+" gap: "+deci.format(getGap())+"%"
				+" K: "+solution.numRoutes
				+" iteration: "+iterator
				+" eta: "+deci.format(acceptanceCriterion.getEta())
				+" omega: "+deci.format(selectedPerturbation.omega)
				+" time: "+timeAF




			
		System.out.println("File: "+file);
		System.out.println("Rounded: "+rounded);
		System.out.println("limit: "+limit);
		System.out.println("Best: "+best);
		System.out.println("LimitTime: "+limit);
		System.out.println(config);