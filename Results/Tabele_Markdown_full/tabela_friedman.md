# Wyniki testu Friedmana

| Zbiór      | Model   |   Liczba metod |   Statystyka χ² | p-value   | Istotność   |
|:-----------|:--------|---------------:|----------------:|:----------|:------------|
| diabetes   | MLP     |              7 |          143.6  | < 0.0001  | **TAK**     |
| diabetes   | NB      |              7 |          147.96 | < 0.0001  | **TAK**     |
| diabetes   | RF      |              7 |           82.27 | < 0.0001  | **TAK**     |
| diabetes   | XGBoost |              7 |           75.42 | < 0.0001  | **TAK**     |
| kredyty    | MLP     |              7 |          124.17 | < 0.0001  | **TAK**     |
| kredyty    | NB      |              7 |          188.76 | < 0.0001  | **TAK**     |
| kredyty    | RF      |              7 |           11.13 | 0.0844    | NIE         |
| kredyty    | XGBoost |              7 |           63.66 | < 0.0001  | **TAK**     |
| rezygnacje | MLP     |              7 |          188.96 | < 0.0001  | **TAK**     |
| rezygnacje | NB      |              7 |          179.47 | < 0.0001  | **TAK**     |
| rezygnacje | RF      |              7 |           37.63 | < 0.0001  | **TAK**     |
| rezygnacje | XGBoost |              7 |           56.45 | < 0.0001  | **TAK**     |
| serce      | MLP     |              7 |          200.58 | < 0.0001  | **TAK**     |
| serce      | NB      |              7 |          112.47 | < 0.0001  | **TAK**     |
| serce      | RF      |              7 |           47.53 | < 0.0001  | **TAK**     |
| serce      | XGBoost |              7 |            6.32 | 0.3887    | NIE         |
| zapalenia  | MLP     |              7 |          208.59 | < 0.0001  | **TAK**     |
| zapalenia  | NB      |              7 |          209.89 | < 0.0001  | **TAK**     |
| zapalenia  | RF      |              7 |          102.72 | < 0.0001  | **TAK**     |
| zapalenia  | XGBoost |              7 |           60.58 | < 0.0001  | **TAK**     |