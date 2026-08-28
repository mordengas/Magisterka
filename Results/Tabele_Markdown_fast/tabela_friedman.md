# Wyniki testu Friedmana

| Zbiór      | Model   |   Liczba metod |   Statystyka χ² | p-value   | Istotność   |
|:-----------|:--------|---------------:|----------------:|:----------|:------------|
| diabetes   | NB      |              5 |            2    | 0.7358    | NIE         |
| diabetes   | RF      |              5 |            6.91 | 0.1406    | NIE         |
| diabetes   | XGBoost |              5 |            0.77 | 0.9425    | NIE         |
| kredyty    | NB      |              5 |            2.47 | 0.6506    | NIE         |
| kredyty    | RF      |              5 |            3.93 | 0.4152    | NIE         |
| kredyty    | XGBoost |              5 |            6.07 | 0.1937    | NIE         |
| rezygnacje | NB      |              5 |           24.56 | < 0.0001  | **TAK**     |
| rezygnacje | RF      |              5 |           11.53 | 0.0212    | **TAK**     |
| rezygnacje | XGBoost |              5 |            0.59 | 0.9639    | NIE         |
| serce      | NB      |              5 |            4.95 | 0.2926    | NIE         |
| serce      | RF      |              5 |            4.51 | 0.3418    | NIE         |
| serce      | XGBoost |              5 |            0.74 | 0.9462    | NIE         |
| zapalenia  | NB      |              5 |           31.6  | < 0.0001  | **TAK**     |
| zapalenia  | RF      |              5 |           29.18 | < 0.0001  | **TAK**     |
| zapalenia  | XGBoost |              5 |           18    | 0.0012    | **TAK**     |