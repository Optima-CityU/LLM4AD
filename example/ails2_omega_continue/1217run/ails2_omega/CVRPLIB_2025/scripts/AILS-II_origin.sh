#!/bin/bash

cd src/AILS-II_origin

mkdir out\production\AILS-II
javac -d out/production/AILS-II -cp src $(find src -name "*.java")
jar cvfe AILSII.jar SearchMethod.AILSII -C out/production/AILS-II .

cd ../../
cp src/AILS-II_origin/AILSII.jar bin/AILSII_origin.jar