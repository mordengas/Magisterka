# Uzasadnienie Metodologiczne Doboru Strategii Czyszczenia Danych

> **Fragment do wykorzystania w rozdziale metodycznym pracy magisterskiej**  
> *(Rozdział: Projekt eksperymentu / Dobór strategii przetwarzania wstępnego)*

---

## 1. Trzy Filary Przetwarzania Wstępnego

W badaniach nad jakością klasyfikatorów tabelarycznych wyodrębniono trzy fundamentalne operacje inżynierii danych:
1. **$O_1$ – Eliminacja i obsługa wartości odstających (*Outlier Removal / Replacement*):** Identyfikacja anomalii metodą rozstępu międzykwartylowego (IQR) i ich konwersja do braków danych (`NaN`).
2. **$O_2$ – Imputacja braków danych (*Missing Value Imputation*):** Rekonstrukcja brakujących wartości za pomocą estymatorów statystycznych (mediana) lub algorytmów uczenia maszynowego ($k$-Nearest Neighbors, $k=5$).
3. **$O_3$ – Standaryzacja cech numerycznych (*Feature Scaling*):** Przekształcenie $z$-score ($\mu=0, \sigma=1$) za pomocą `StandardScaler` wraz z binarnym kodowaniem cech kategorycznych (One-Hot Encoding).

---

## 2. Redukcja Pełnego Planu Czynnikowego ($2^3 = 8$) do Zestawu Strategii Praktycznych

Początkowa wersja eksperymentu zakładała pełny plan kombinatoryczny $2^3 = 8$ metod:
$$\{\text{raw}, \text{norm}, \text{fill}, \text{remove}, \text{remove\_fill}, \text{remove\_norm}, \text{fill\_norm}, \text{all}\}$$

W toku analizy teoretycznej oraz weryfikacji eksperymentalnej zidentyfikowano istotne wady metod **pozbawionych kroku imputacji** ($\text{fill}$):

### A. Problem sztucznej stałej w metodzie `remove`
* W operacji `remove` wykryte wartości odstające zamieniane są na `NaN`.
* Brak etapu rekonstrukcji braków zmusza algorytm do zastąpienia braków techniczną wartością stałą (np. $-999.0$), aby umożliwić estymację parametrów modeli nieobsługujących braków danych.
* **Wniosek:** Procedura ta de facto tworzy nową, skrajną wartość odstającą, negując cel operacji eliminacji anomalii.

### B. Zniekształcenie parametrów rozkładu w metodzie `norm`
* Standaryzacja cech numerycznych w obecności nieskorygowanych wartości odstających oraz niezaimputowanych braków prowadzi do błędnej estymacji średniej $\hat{\mu}$ oraz zawyżenia odchylenia standardowego $\hat{\sigma}$.
* W efekcie znormalizowane wartości cech typowych ulegają sztucznej kompresji blisko zera.

---

## 3. Nowy Zestaw Strategii Badawczych (Profil Rozszerzony)

Zamiast analizować kombinacje syntetycznie zaburzające proces uczenia (metody bez imputacji), przestrzeń eksperymentu została zoptymalizowana i rozszerzona o **porównanie paradygmatów imputacji** (statystyczna vs algorytmiczna):

| Kod Strategii | Składowe Pipeline'u | Uzasadnienie i Rola Badawcza |
| :--- | :--- | :--- |
| **`raw`** | Brak operacji (maskowanie -999) | Punkt odniesienia (*baseline*) – degradacja modelu bez interwencji. |
| **`fill`** | Imputacja medianą | Wpływ samej prostej rekonstrukcji braków. |
| **`fill_knn`** | Imputacja $k$-NN ($k=5$) | Wpływ wielowymiarowej rekonstrukcji braków opartej na odległościach. |
| **`remove_fill`** | Outliery (IQR) $\to$ Imputacja medianą | Dwuetapowa korekcja: oczyszczenie rozkładu przed estymacją mediany. |
| **`fill_norm`** | Imputacja medianą $\to$ Standaryzacja | Wpływ skalowania na zaimputowanych danych. |
| **`all`** | Outliery $\to$ Mediana $\to$ Standaryzacja | Pełny klasyczny pipeline inżynierii danych. |
| **`all_knn`** | Outliery $\to$ $k$-NN $\to$ Standaryzacja | Zaawansowany pipeline bazujący na modelowaniu najbliższych sąsiadów. |

---

## 4. Główne Pytania Badawcze Wynikające z Nowego Układu

1. **Efekt synergii:** Czy sekwencja $\text{Outliery} \to \text{Imputacja} \to \text{Skalowanie}$ (`all`) daje zysk większy niż suma pojedynczych kroków?
2. **Opłacalność obliczeniowa $k$-NN (*Computational Trade-off*):** Czy narzut czasowy związany z wyznaczaniem odległości w $k$-NN ($10\times - 50\times$ dłuższy czas) przekłada się na statystycznie istotny wzrost AUC względem mediany?
3. **Odporność architektur:** W jakim stopniu modele odporne z natury (drzewa decyzyjne XGBoost/RF) różnią się w zyskach z preprocessingu od modeli wrażliwych (Naiwny Bayes, Sieci MLP)?
