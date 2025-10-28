#!/bin/bash

time=(13500 14460)
best=(365 1234)
instances=(XLTEST-n5649-k365 XLTEST-n6034-k1234)

mkdir -p remote_results/AILSII

for i in {0..7}; do
    inst=${instances[$i]}

    for j in {1..1}; do     
        echo "Run $j instance $inst"
        java -jar -Xms2000m -Xmx4000m bin/AILSII.jar \
             -file XLDemo/${inst}.vrp \
             -rounded true \
             -stoppingCriterion Time \
             -limit ${time[$i]} \
             -best ${best[$i]} \
        > remote_results/AILSII/${inst}.csv

        # 添加日志信息
        echo "Completed run $j for instance $inst, results saved to remote_results/AILSII/${inst}.csv"
    done
done