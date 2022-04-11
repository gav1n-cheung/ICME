# ICME 数据可视化工具
依赖
```
pip install -r install.md
```

需要编译coco API

```
python data_summary.py
```
coco_train_2017输出结果：
![result table](https://github.com/gav1n-cheung/ICME/blob/main/res/coco.png)
```
==============image sum:118287==============
+----------------+---------------------+
|     class      |  small object ratio |
+----------------+---------------------+
|     person     |  0.6105961468556889 |
|    bicycle     |  0.6930379746835443 |
|      car       |  0.8555900621118012 |
|   motorcycle   | 0.48247978436657685 |
|    airplane    | 0.34265734265734266 |
|      bus       |  0.3508771929824561 |
|     train      | 0.07894736842105263 |
|     truck      |  0.5975903614457831 |
|      boat      |  0.7441860465116279 |
| traffic light  |  0.9544740973312402 |
|  fire hydrant  | 0.46534653465346537 |
|   stop sign    |         0.6         |
| parking meter  |  0.5333333333333333 |
|     bench      |  0.5907990314769975 |
|      bird      |  0.7818181818181819 |
|      cat       | 0.08415841584158416 |
|      dog       | 0.22935779816513763 |
|     horse      |  0.3992673992673993 |
|     sheep      |  0.5789473684210527 |
|      cow       |  0.6368421052631579 |
|    elephant    |  0.3137254901960784 |
|      bear      |  0.2112676056338028 |
|     zebra      |  0.3694029850746269 |
|    giraffe     | 0.21982758620689655 |
|    backpack    |  0.8463611859838275 |
|    umbrella    |  0.5786924939467313 |
|    handbag     |  0.8462962962962963 |
|      tie       |  0.7480314960629921 |
|    suitcase    |  0.6138613861386139 |
|    frisbee     |  0.8434782608695652 |
|      skis      |  0.7759336099585062 |
|   snowboard    |  0.7536231884057971 |
|  sports ball   |  0.9657794676806084 |
|      kite      |  0.7708333333333334 |
|  baseball bat  |  0.6712328767123288 |
| baseball glove |  0.9527027027027027 |
|   skateboard   |  0.664804469273743  |
|   surfboard    |  0.6840148698884758 |
| tennis racket  |  0.6977777777777778 |
|     bottle     |  0.8692682926829268 |
|   wine glass   |  0.8017492711370262 |
|      cup       |  0.8242491657397107 |
|      fork      |  0.6139534883720931 |
|     knife      |  0.8190184049079755 |
|     spoon      |  0.7944664031620553 |
|      bowl      |  0.6277955271565495 |
|     banana     |  0.6649076517150396 |
|     apple      |  0.7531380753138075 |
|    sandwich    |  0.3050847457627119 |
|     orange     |  0.662020905923345  |
|    broccoli    |  0.620253164556962  |
|     carrot     |  0.8005390835579514 |
|    hot dog     |  0.5275590551181102 |
|     pizza      | 0.39649122807017545 |
|     donut      |  0.6863905325443787 |
|      cake      |  0.5569620253164557 |
|     chair      |  0.6806253489670575 |
|     couch      | 0.11494252873563218 |
|  potted plant  |  0.6618075801749271 |
|      bed       | 0.03680981595092025 |
|  dining table  | 0.27259684361549497 |
|     toilet     | 0.16759776536312848 |
|       tv       |       0.40625       |
|     laptop     | 0.31601731601731603 |
|     mouse      |  0.8584905660377359 |
|     remote     |  0.9187279151943463 |
|    keyboard    |  0.4444444444444444 |
|   cell phone   |  0.7938931297709924 |
|   microwave    |  0.5636363636363636 |
|      oven      | 0.25874125874125875 |
|    toaster     |  0.7777777777777778 |
|      sink      |  0.5777777777777777 |
|  refrigerator  | 0.10317460317460317 |
|      book      |  0.8914728682170543 |
|     clock      |  0.7865168539325843 |
|      vase      |  0.7509025270758123 |
|    scissors    |  0.4444444444444444 |
|   teddy bear   |  0.3507853403141361 |
|   hair drier   |  0.6363636363636364 |
|   toothbrush   |  0.7719298245614035 |
+----------------+---------------------+
+------------------+-----+
| bbox scale ratio | sum |
+------------------+-----+
|        2         |  28 |
|        1         |  29 |
|        3         |  7  |
|        6         |  3  |
|        5         |  5  |
|        4         |  2  |
|        8         |  3  |
|        7         |  2  |
|        12        |  1  |
+------------------+-----+
```
