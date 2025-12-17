# CVRPLIB-2025

# 编译
```
sudo chmod +x ./scripts/*
./scripts/compile.sh
sudo chmod +x ./bin/* 
```


[//]: # (```)

[//]: # (python run.py AILSII_CPU.jar)

[//]: # (python run.py hgs-TV)

[//]: # (python run.py filo2)

[//]: # (```)

测试命令（仅跑前10个实例，每个方法限时10秒）
```
python run.py AILSII_origin.jar --start-idx 0 --end-idx 10 --time-limit 10
python run.py AILSII_perturbation1.jar --start-idx 0 --end-idx 10 --time-limit 10
python run.py AILSII_perturbation2.jar --start-idx 0 --end-idx 10 --time-limit 10
python run.py AILSII_deco.jar --start-idx 0 --end-idx 10 --time-limit 10
```
# 请在不同服务器上分别运行以下命令
```
python run.py AILSII_origin.jar --start-idx 0 --end-idx 50
```

```
python run.py AILSII_origin.jar --start-idx 50 --end-idx 100
```

```
python run.py AILSII_perturbation1.jar --start-idx 0 --end-idx 50
```

```
python run.py AILSII_perturbation1.jar --start-idx 50 --end-idx 100
```

```
python run.py AILSII_perturbation2.jar --start-idx 0 --end-idx 50
```

```
python run.py AILSII_perturbation2.jar --start-idx 50 --end-idx 100
```

```
python run.py AILSII_deco.jar --start-idx 0 --end-idx 50
```

```
python run.py AILSII_deco.jar --start-idx 50 --end-idx 100
```

# 测试EoH版本，请在不同服务器上分别运行以下命令
100个instances：在不同服务器上分别运行以下命令
```
python run.py AILSII_EoH.jar --start-idx 0 --end-idx 50
```

```
python run.py AILSII_EoH.jar --start-idx 50 --end-idx 100
```
8个instances
```
python run_ails2_parallel.py
```


