# Wpływ metod czyszczenia danych na jakość klasyfikatorów

Projekt badawczy do pracy magisterskiej badający wpływ różnych strategii wstępnego przetwarzania i czyszczenia danych na jakość klasyfikacji modeli uczenia maszynowego w warunkach kontrolowanej degradacji danych.

---

## 📁 Struktura Projektu

```text
├── config.py                  # Główna konfiguracja: definicje zbiorów, hiperparametry modeli, profile eksperymentów
├── run_pipeline.py            # Główny runner całego pipeline eksperymentalnego
├── requirements.txt           # Zależności bibliotek Python
├── Data/                      # Zbiory danych (oryginalne oraz generowane uszkodzone)
│   ├── zapalenia/
│   ├── diabetes/
│   ├── serce/
│   ├── rezygnacje/
│   └── kredyty/
├── Src/
│   ├── cleaning_methods.py          # Custom scikit-learn transformers i budowa pipeline'ów czyszczenia
│   ├── stworz_problemy.py           # Zunifikowany generator uszkodzeń (braki danych, outliery, szum kategoryczny)
│   ├── testy_statystyczne.py        # Sparowane testy Wilcoxona, testy Friedmana i diagramy CD (Critical Difference)
│   ├── wizualizacja.py              # Podstawowe wykresy słupkowe i rozkłady AUC
│   ├── wizualizacje_zaawansowane.py # Zaawansowane wykresy (krzywe odporności, heatmapy Delta AUC, radary, cechy)
│   ├── generuj_tabele_latex.py      # Automatyczne generowanie tabel LaTeX i Markdown z wyników
│   ├── naprawienie_problemow.py     # Skrypt pomocniczy do inspekcji danych po wyczyszczeniu
│   └── sprawdz_waznosc.py           # Szybka analiza ważności cech (Random Forest Gini importance)
├── KlasaTestowa/
│   └── TestMyClassifier.py          # Silnik ewaluacji: Stratified K-Fold CV bez data leakage z paralelizacją joblib
├── Results/                         # Wyniki ewaluacji (CSV, tabele LaTeX, wykresy PNG)
│   ├── Tabele_LaTeX_10_50/
│   ├── Tabele_Markdown_10_50/
│   ├── Wykresy_10_50/
│   └── Wykresy_Zaawansowane/
└── Magisterka-latex/                # Źródła pracy magisterskiej w LaTeX
```

---

## 🚀 Uruchamianie Eksperymentów

Projekt obsługuje dwa profile eksperymentalne:
- **`full`** (profil główny): 5 poziomów uszkodzeń (10%, 20%, 30%, 40%, 50%), 7 metod czyszczenia (w tym imputacja KNN), 4 klasyfikatory (`RF`, `NB`, `MLP`, `XGBoost`), 3 powtórzenia uszkodzeń, 4-krotna stratyfikowana walidacja krzyżowa (CV) z 3 powtórzeniami seedów.
- **`fast`** (profil szybki/testowy): 3 poziomy uszkodzeń (20%, 40%, 60%), 5 podstawowych metod czyszczenia, 3 klasyfikatory (`RF`, `NB`, `XGBoost`), 2 powtórzenia uszkodzeń, 3-krotna CV.

### 1. Uruchomienie pełnego pipeline jednym poleceniem:
```bash
python run_pipeline.py --profile full
```
lub dla szybkiego testu:
```bash
python run_pipeline.py --profile fast
```

### 2. Uruchamianie poszczególnych kroków ręcznie:
```bash
# Krok 1: Generowanie kontrolowanych uszkodzeń w danych
python Src/stworz_problemy.py --profile full

# Krok 2: Walidacja krzyżowa modeli bez wycieku danych (Data Leakage)
python KlasaTestowa/TestMyClassifier.py --profile full

# Krok 3: Generowanie podstawowych wykresów AUC
python Src/wizualizacja.py --profile full

# Krok 4: Generowanie zaawansowanych wizualizacji
python Src/wizualizacje_zaawansowane.py --profile full

# Krok 5: Obliczenia testów statystycznych (Wilcoxon, Friedman, diagramy CD)
python Src/testy_statystyczne.py --profile full

# Krok 6: Generowanie tabel LaTeX i Markdown
python Src/generuj_tabele_latex.py --profile full
```

---

## 🧹 Badane Metody Czyszczenia Danych

1. `raw` — brak naprawy (dane uszkodzone ze standardową obsługą braków).
2. `fill` — uzupełnienie braków danych medianą (cechy ciągłe) i dominantą (cechy kategoryczne).
3. `fill_knn` — algorytmiczne uzupełnianie braków danych za pomocą $k$-najbliższych sąsiadów ($k=5$).
4. `remove_fill` — wykrycie i usunięcie wartości odstających (IQR $\times 3.0$) oraz rzadkich kategorii ($<5\%$), a następnie uzupełnienie medianą / dominantą.
5. `fill_norm` — uzupełnienie braków danych medianą + standaryzacja cech ciągłych (`StandardScaler`).
6. `all` — kompleksowy pipeline: usuwanie anomalii + imputacja medianą/dominantą + standaryzacja cech.
7. `all_knn` — kompleksowy pipeline: usuwanie anomalii + zaawansowana imputacja KNN + standaryzacja cech.
