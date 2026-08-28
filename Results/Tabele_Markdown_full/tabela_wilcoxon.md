# Wyniki testu Wilcoxona (Metoda vs RAW)

| Zbiór      | Model   | Metoda      |   n |   Statystyka W | p-value   | Istotność   |
|:-----------|:--------|:------------|----:|---------------:|:----------|:------------|
| diabetes   | MLP     | all         |  45 |           11   | < 0.0001  | **TAK ***** |
| diabetes   | MLP     | all_knn     |  45 |            0   | < 0.0001  | **TAK ***** |
| diabetes   | MLP     | fill        |  45 |          517   | 1.0000    | NIE         |
| diabetes   | MLP     | fill_knn    |  45 |          418.5 | 0.2638    | NIE         |
| diabetes   | MLP     | fill_norm   |  45 |          317   | 0.0378    | **TAK ***   |
| diabetes   | MLP     | remove_fill |  45 |           69   | < 0.0001  | **TAK ***** |
| diabetes   | NB      | all         |  45 |           27   | < 0.0001  | **TAK ***** |
| diabetes   | NB      | all_knn     |  45 |           17.5 | < 0.0001  | **TAK ***** |
| diabetes   | NB      | fill        |  45 |           61   | < 0.0001  | **TAK ***** |
| diabetes   | NB      | fill_knn    |  45 |          486   | 0.7222    | NIE         |
| diabetes   | NB      | fill_norm   |  45 |           61   | < 0.0001  | **TAK ***** |
| diabetes   | NB      | remove_fill |  45 |           27   | < 0.0001  | **TAK ***** |
| diabetes   | RF      | all         |  45 |          111   | < 0.0001  | **TAK ***** |
| diabetes   | RF      | all_knn     |  45 |          430   | 0.3233    | NIE         |
| diabetes   | RF      | fill        |  45 |          197.5 | 0.0003    | **TAK ***** |
| diabetes   | RF      | fill_knn    |  45 |          507.5 | 0.9101    | NIE         |
| diabetes   | RF      | fill_norm   |  45 |          189.5 | 0.0002    | **TAK ***** |
| diabetes   | RF      | remove_fill |  45 |          116   | < 0.0001  | **TAK ***** |
| diabetes   | XGBoost | all         |  45 |          135   | < 0.0001  | **TAK ***** |
| diabetes   | XGBoost | all_knn     |  45 |          238   | 0.0016    | **TAK ****  |
| diabetes   | XGBoost | fill        |  45 |          300.5 | 0.0232    | **TAK ***   |
| diabetes   | XGBoost | fill_knn    |  45 |          239   | 0.0017    | **TAK ****  |
| diabetes   | XGBoost | fill_norm   |  45 |          300.5 | 0.0232    | **TAK ***   |
| diabetes   | XGBoost | remove_fill |  45 |          135   | < 0.0001  | **TAK ***** |
| kredyty    | MLP     | all         |  45 |           24   | < 0.0001  | **TAK ***** |
| kredyty    | MLP     | all_knn     |  45 |           35   | < 0.0001  | **TAK ***** |
| kredyty    | MLP     | fill        |  45 |          415   | 0.2473    | NIE         |
| kredyty    | MLP     | fill_knn    |  45 |          437   | 0.3635    | NIE         |
| kredyty    | MLP     | fill_norm   |  45 |          210   | 0.0003    | **TAK ***** |
| kredyty    | MLP     | remove_fill |  45 |          386   | 0.1377    | NIE         |
| kredyty    | NB      | all         |  45 |            2   | < 0.0001  | **TAK ***** |
| kredyty    | NB      | all_knn     |  45 |            2   | < 0.0001  | **TAK ***** |
| kredyty    | NB      | fill        |  45 |          364.5 | 0.1278    | NIE         |
| kredyty    | NB      | fill_knn    |  45 |          511.5 | 0.9460    | NIE         |
| kredyty    | NB      | fill_norm   |  45 |            3   | < 0.0001  | **TAK ***** |
| kredyty    | NB      | remove_fill |  45 |            0   | < 0.0001  | **TAK ***** |
| kredyty    | RF      | all         |  45 |          289   | 0.0099    | **TAK ****  |
| kredyty    | RF      | all_knn     |  45 |          347   | 0.0544    | NIE         |
| kredyty    | RF      | fill        |  45 |          369   | 0.0937    | NIE         |
| kredyty    | RF      | fill_knn    |  45 |          468.5 | 0.5802    | NIE         |
| kredyty    | RF      | fill_norm   |  45 |          372.5 | 0.1017    | NIE         |
| kredyty    | RF      | remove_fill |  45 |          271   | 0.0089    | **TAK ****  |
| kredyty    | XGBoost | all         |  45 |          196   | 0.0003    | **TAK ***** |
| kredyty    | XGBoost | all_knn     |  45 |          460   | 0.5163    | NIE         |
| kredyty    | XGBoost | fill        |  45 |          247   | 0.0023    | **TAK ****  |
| kredyty    | XGBoost | fill_knn    |  45 |          425   | 0.2964    | NIE         |
| kredyty    | XGBoost | fill_norm   |  45 |          247   | 0.0023    | **TAK ****  |
| kredyty    | XGBoost | remove_fill |  45 |          196   | 0.0003    | **TAK ***** |
| rezygnacje | MLP     | all         |  45 |            1   | < 0.0001  | **TAK ***** |
| rezygnacje | MLP     | all_knn     |  45 |            5   | < 0.0001  | **TAK ***** |
| rezygnacje | MLP     | fill        |  45 |          510.5 | 0.9370    | NIE         |
| rezygnacje | MLP     | fill_knn    |  45 |          415   | 0.2523    | NIE         |
| rezygnacje | MLP     | fill_norm   |  45 |           93   | < 0.0001  | **TAK ***** |
| rezygnacje | MLP     | remove_fill |  45 |            6   | < 0.0001  | **TAK ***** |
| rezygnacje | NB      | all         |  45 |           23   | < 0.0001  | **TAK ***** |
| rezygnacje | NB      | all_knn     |  45 |           24   | < 0.0001  | **TAK ***** |
| rezygnacje | NB      | fill        |  45 |          473.5 | 0.6193    | NIE         |
| rezygnacje | NB      | fill_knn    |  45 |          264.5 | 0.0043    | **TAK ****  |
| rezygnacje | NB      | fill_norm   |  45 |            0   | < 0.0001  | **TAK ***** |
| rezygnacje | NB      | remove_fill |  45 |           24   | < 0.0001  | **TAK ***** |
| rezygnacje | RF      | all         |  45 |          499.5 | 0.8390    | NIE         |
| rezygnacje | RF      | all_knn     |  45 |          353.5 | 0.0987    | NIE         |
| rezygnacje | RF      | fill        |  45 |          411.5 | 0.2315    | NIE         |
| rezygnacje | RF      | fill_knn    |  45 |           12   | < 0.0001  | **TAK ***** |
| rezygnacje | RF      | fill_norm   |  45 |          415.5 | 0.2496    | NIE         |
| rezygnacje | RF      | remove_fill |  45 |          495.5 | 0.8039    | NIE         |
| rezygnacje | XGBoost | all         |  45 |          369   | 0.0937    | NIE         |
| rezygnacje | XGBoost | all_knn     |  45 |          139   | < 0.0001  | **TAK ***** |
| rezygnacje | XGBoost | fill        |  45 |          300.5 | 0.0143    | **TAK ***   |
| rezygnacje | XGBoost | fill_knn    |  45 |           45.5 | < 0.0001  | **TAK ***** |
| rezygnacje | XGBoost | fill_norm   |  45 |          300.5 | 0.0143    | **TAK ***   |
| rezygnacje | XGBoost | remove_fill |  45 |          369   | 0.0937    | NIE         |
| serce      | MLP     | all         |  45 |            0   | < 0.0001  | **TAK ***** |
| serce      | MLP     | all_knn     |  45 |            0   | < 0.0001  | **TAK ***** |
| serce      | MLP     | fill        |  45 |          407   | 0.2166    | NIE         |
| serce      | MLP     | fill_knn    |  45 |          407.5 | 0.2144    | NIE         |
| serce      | MLP     | fill_norm   |  45 |            0   | < 0.0001  | **TAK ***** |
| serce      | MLP     | remove_fill |  45 |          474.5 | 0.6274    | NIE         |
| serce      | NB      | all         |  45 |          364.5 | 0.1278    | NIE         |
| serce      | NB      | all_knn     |  45 |          330.5 | 0.0348    | **TAK ***   |
| serce      | NB      | fill        |  45 |          129   | < 0.0001  | **TAK ***** |
| serce      | NB      | fill_knn    |  45 |          249.5 | 0.0025    | **TAK ****  |
| serce      | NB      | fill_norm   |  45 |           11   | < 0.0001  | **TAK ***** |
| serce      | NB      | remove_fill |  45 |          331   | 0.0556    | NIE         |
| serce      | RF      | all         |  45 |          162   | < 0.0001  | **TAK ***** |
| serce      | RF      | all_knn     |  45 |          215.5 | 0.0007    | **TAK ***** |
| serce      | RF      | fill        |  45 |          245.5 | 0.0021    | **TAK ****  |
| serce      | RF      | fill_knn    |  45 |          243   | 0.0015    | **TAK ****  |
| serce      | RF      | fill_norm   |  45 |          250   | 0.0025    | **TAK ****  |
| serce      | RF      | remove_fill |  45 |          169   | < 0.0001  | **TAK ***** |
| serce      | XGBoost | all         |  45 |          395   | 0.1667    | NIE         |
| serce      | XGBoost | all_knn     |  45 |          318   | 0.0243    | **TAK ***   |
| serce      | XGBoost | fill        |  45 |          505.5 | 0.8923    | NIE         |
| serce      | XGBoost | fill_knn    |  45 |          353   | 0.0633    | NIE         |
| serce      | XGBoost | fill_norm   |  45 |          505.5 | 0.8923    | NIE         |
| serce      | XGBoost | remove_fill |  45 |          395   | 0.1667    | NIE         |
| zapalenia  | MLP     | all         |  45 |            0   | < 0.0001  | **TAK ***** |
| zapalenia  | MLP     | all_knn     |  45 |            0   | < 0.0001  | **TAK ***** |
| zapalenia  | MLP     | fill        |  45 |          491   | 0.7648    | NIE         |
| zapalenia  | MLP     | fill_knn    |  45 |          344   | 0.0780    | NIE         |
| zapalenia  | MLP     | fill_norm   |  45 |            0   | < 0.0001  | **TAK ***** |
| zapalenia  | MLP     | remove_fill |  45 |          407   | 0.2166    | NIE         |
| zapalenia  | NB      | all         |  45 |            0   | < 0.0001  | **TAK ***** |
| zapalenia  | NB      | all_knn     |  45 |            0   | < 0.0001  | **TAK ***** |
| zapalenia  | NB      | fill        |  45 |          245   | 0.0162    | **TAK ***   |
| zapalenia  | NB      | fill_knn    |  45 |          126   | < 0.0001  | **TAK ***** |
| zapalenia  | NB      | fill_norm   |  45 |           93   | < 0.0001  | **TAK ***** |
| zapalenia  | NB      | remove_fill |  45 |           16   | < 0.0001  | **TAK ***** |
| zapalenia  | RF      | all         |  45 |           92.5 | < 0.0001  | **TAK ***** |
| zapalenia  | RF      | all_knn     |  45 |           96   | < 0.0001  | **TAK ***** |
| zapalenia  | RF      | fill        |  45 |          292.5 | 0.0111    | **TAK ***   |
| zapalenia  | RF      | fill_knn    |  45 |          440   | 0.3817    | NIE         |
| zapalenia  | RF      | fill_norm   |  45 |          299   | 0.0136    | **TAK ***   |
| zapalenia  | RF      | remove_fill |  45 |           94   | < 0.0001  | **TAK ***** |
| zapalenia  | XGBoost | all         |  45 |          263   | 0.0035    | **TAK ****  |
| zapalenia  | XGBoost | all_knn     |  45 |          197   | 0.0002    | **TAK ***** |
| zapalenia  | XGBoost | fill        |  45 |          390   | 0.1501    | NIE         |
| zapalenia  | XGBoost | fill_knn    |  45 |          462.5 | 0.5347    | NIE         |
| zapalenia  | XGBoost | fill_norm   |  45 |          390   | 0.1501    | NIE         |
| zapalenia  | XGBoost | remove_fill |  45 |          263   | 0.0035    | **TAK ****  |