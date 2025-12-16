#!/bin/bash

cd src/AILS-II_deco

mkdir out\production\AILS-II
javac -d out/production/AILS-II -cp src $(find src -name "*.java")
jar cvfe AILSII.jar SearchMethod.AILSII -C out/production/AILS-II .

cd ../../
cp src/AILS-II_deco/AILSII.jar bin/AILSII_deco.jar