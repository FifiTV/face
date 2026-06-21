# Raport z projektu grupowego – Uwierzytelnianie *in-the-wild*

---

## Dane autorów

| | |
|-|-|
| **Imię i nazwisko** | Jakub Drzewiecki | 
| **Numer indeksu** | 268418 | 
| **Imię i nazwisko** | Filip Fit | 
| **Numer indeksu** | 288773 | 
| **Termin laboratorium** | Wtorek 7:30 | 
| **Data oddania** | 26.05.2026 | 
---

## 1. Problem

Podstawowym problemem rozważanym w projekcie jest uwierzytelnianie użytkowników na podstawie fotografii twarzy. Mając taki system w wersji podstawowej, chcielbyśmy teraz zweryfikować jego działanie w przypadku dwóch utrudnień:

- **Wiele twarzy na zdjęciu** – weryfikacja, czy system potrafi poprawnie zidentyfikować użytkownika, gdy na zdjęciu znajduje się więcej niż jedna twarz. Porządane jest, aby system nie weryfikował zdjęć, na których twarz użytkownika jest tylko w tle - zakładamy, że próby weryfikacji podejmuje osoba w centrum zdjęcia.
- **Zdjęcie o niskiej rozdzielczości** – weryfikacja, czy system potrafi poprawnie zidentyfikować użytkownika, gdy zdjęcie jest niskiej jakości. Porządane jest, aby system nie weryfikował zdjęć, na których twarz użytkownika jest niewyraźna lub rozmyta.

---

## 2. Opis metody autoryzacji

### 2.1 Model bazowy
System wykorzystuje model **ArcFace ResNet-50** ( `buffalo_l` ) z biblioteki [InsightFace](https://github.com/deepinsight/insightface). Model
generuje 512-wymiarowe embeddingi twarzy normalizowane do jednostkowej normy L2. Detekcja
twarzy realizowana jest przez **RetinaFace** wbudowany w pipeline InsightFace.
ArcFace wprowadza stratę Additive Angular Margin (arcface loss), która maksymalizuje kąt między
klasami w przestrzeni cech, co przekłada się na lepszą separowalność embeddingów niż
tradycyjna softmax loss [Deng et al., 2019].

### 2.2 Baza danych embeddingów
Embeddingi przechowywane są w **ChromaDB** z indeksem HNSW (cosine similarity). Dla każdego
użytkownika podczas enrollowania obliczany jest osobny embedding dla każdego zdjęcia (bez
uśredniania), co pozwala indeksowi HNSW znaleźć najlepiej dopasowany prototyp.

### 2.3 Protokół autoryzacji
**Weryfikacja 1:1**: embedding próbki porównywany z max cosine similarity względem
wszystkich embeddingów danego użytkownika; wynik >= progu -> akceptacja.  
**Identyfikacja 1:N**: HNSW zwraca globalnie najbliższy embedding wraz z ID
właściciela i wynikiem podobieństwa.

### 2.4 Modyfikacje wykonane pod kątem utrudnień

1. Ważony wybór twarzy ze zdjęcia na podstawie rozmiaru i położenia w kadrze. Waga to 0.7 * (f_a / p_a) + 0.3 * d, gdzie f_a to powierzchnia twarzy, p_a to powierzchnia zdjęcia, a d to odległości centrum twarzy od środka obrazu.
2. Zwiększenie rozdzielczości zdjęć niskiej jakości za pomocą modelu **Real-ESRGAN** (x2 - dwukrotne powiększenie rozdzielczości). Tylko dla TinyFace ze względu na czas przetwarzania.

---

## 3. Opis danych
### 3.1 FaceScrub
Wykorzystany jako baseline oraz do problemu autoryzacji z wieloma twarzami na zdjęciu. Zdjęcia są niejednorodne pod względem oświetlenia, kąta i rozdzielczości, twarze wykryte wcześniej i przycięte. 

- Rozdzielczość: zróżnicowana (FaceScrub dostarcza już przycięte twarze)
- Kanał kolorów: BGR/RGB
- Format: JPEG
- Liczba zdjęć na osobę: 3–89, średnio 53,5 (train); 2–38, średnio 22,9 (test)

### 3.2 TinyFace
Wykorzystany do problemu autoryzacji z niską rozdzielczością zdjęć. Zbiór z góry podzielony jest na podzbiory treningowy i testowy. Zdjęcia są niskiej jakości, często rozmyte, z małymi twarzami.

- Rozdzielczość: średnio 20x16 pikseli
- Kanał kolorów: RGB
- Format: JPG
- Liczba zdjęć na osobę: 1-29, średnio 1,73 (train); 1-29, średnio 1,45 (test)

## 4. Procedura enrollowania i autoryzacji
### 4.1 Enrollowanie
1. Dla każdej osoby z katalogu `data/enrolled/<name>/` ładowane są wszystkie zdjęcia
JPEG/PNG.
2. Każde zdjęcie przechodzi przez pipeline InsightFace (detekcja + alignment).
3. Jeśli detekcja zawiedzie (zdjęcie już przycięte), stosowany jest tryb fallback:
bezpośrednie wywołanie rec.get_feat([img]) .
4. Embedding zapisywany jest do ChromaDB.

### 4.2 Procedura testowa
- **Wiele twarzy na zdjęciu**: zdjęcia generowane były na podstawie zbioru FaceScrub poprzez łączenie zdjęć różnych osób w jeden obraz (np. 2-3 twarze na zdjęciu). Próbki dzielone były na dwie grupy - z twarzą użytkownika w centrum (pozytywne) i z twarzą użytkownika w tle (negatywne). Twarz wybrana ze zdjęcia porównywana była z embeddingami użytkownika w ChromaDB, a wynik porównania (cosine similarity) był oceniany względem progu decyzyjnego.
- **TinyFace**: zdjęcia z tego zbioru były bezpośrednio porównywane z embeddingami użytkownika w ChromaDB, a wynik porównania (cosine similarity) był oceniany względem progu decyzyjnego.
- **Próbki czyste**: zwykłe zdjęcia z FaceScrub, bez modyfikacji, służące jako kontrola.

Wszystkie testy przeprowadzane były na zamkniętym zbiorze użytkowników jako próby poprawnej autentykacji (10%) oraz próby podszycia się pod użytkownika.

Próg decyzyjny weryfikacji użytkownika - **0.4**.

## 5. Wyniki

### 5.1 Baseline

Eksperyment na zamkniętym zbiorze: wyłącznie osoby enrolled (105 użytkowników, 2 397 genuine + 23 970 impostor par).

| Metryka | Wartość |
|---------|---------|
| FAR | 0.0% |
| FRR | 2.09% |
| Dokładność | 99.81% |
| EER | 0.86% |
| EER Threshold | 0.1842 |
| AUC | 0.9963 |

### 5.2 Wiele twarzy na zdjęciu

Do eksperymentu wygenerowano po 2500 próbek danych pozytywnych (użytkownik w centrum) i negatywnych (użytkownik w tle). Każda próbka wykonana była przy pomocy twarzy losowych osób z FaseScrub. Przykładowe dane:

![multiple_faces](../results/multiple_faces.png)

| Metryka | Wartość |
|---------|---------|
| FAR | 0.08% |
| FRR | 1.36% |
| Dokładność | 99.28% |
| EER | 0.64% |
| AUC | 0.9949 |


### 5.3 TinyFace

W pierwszej kolejności przeprowadzono testy na zbiorze TinyFace bez żadnych modyfikacji zdjęć. Wyniki przedstawiono w tabeli poniżej. Wyłącznie osoby enrolled (2563 użytkoników, 3728 genuine + 37280 impostor par).

| Metryka | Wartość |
|---------|---------|
| FAR | 0.82% |
| FRR | 46.03% |
| Dokładność | 95.07% |
| EER | 14.84% |
| AUC | 0.9267 |

![roc_auc](../results/roc_eer_tiny.png)

W celu poprawy wyników na zbiorze TinyFace, zastosowano model Real-ESRGAN (x2) do zwiększenia rozdzielczości zdjęć. Ze względu na czas przetwarzania, testy przeprowadzono dla tylko 30 użytkowników (87 próbek enrollment, 69 próbek testowych).

Poniżej przedstawiono porównanie zdjęć kilku par zdjęć przed i po zastosowaniu Real-ESRGAN dla tego samego użytkownika.

![enhancement_comparison](../results/comparison_enhancement.png)
![enhancement_comparison](../results/comparison_enhancement2.png)
![enhancement_comparison](../results/comparison_enhancement3.png)
![enhancement_comparison](../results/comparison_enhancement4.png)

| Metryka | Wartość |
|---------|---------|
| FAR | 4.78% |
| FRR | 37.68% |
| Dokładność | 92.23% |
| EER | 12.75% |
| AUC | 0.9178 |

![roc_auc](../results/roc_eer_enhanced_tiny.png)

Niestety, poprawa FRR przyszła kosztem zwiększenia FAR. Real-ESRGAN wprowadza dodatkowe artefakty i zdjęcia tej samej osoby nie do końca przypominają tą samą osobę, co może być bezpośrednim zwiększeniem FAR. 

Spróbujmy teraz enrollować użytkowników na podstawie raw zdjęć, a testować na zdjęciach po enhancement. Próg 0.4 nie jest już optymalny (FAR 0%, FRR 67%), ale po obniżeniu progu do 0.25 otrzymujemy następujące wyniki:

| Metryka | Wartość |
|---------|---------|
| FAR | 2.46% |
| FRR | 18.84% |
| Dokładność | 96.05% |
| EER | 7.83% |
| AUC | 0.9606 |

![roc_auc](../results/roc_eer_enhanced_input_tiny.png)

### 5.4 Wyniki autorów modelu
Model ArcFace (ResNet-50) raportuje w oryginalnej publikacji [Deng et al., 2019]:

| Benchmark | Wartość |
|-|-|
| LFW Accuracy | 99.83% |
| IJB-C TAR@FAR=1% | 96.77% |
| MegaFace Rank-1 | 98.35% |

Nasz baseline jest bardzo bliski wynikowi LFW Accuracy autorów (99.81%), utrudnienie w postaci wielu twarzy zaowocowało uzyskaniu dokładności 99.28%, a dla twarzy o niskiej dokładności wynik wyniósł 96.05%. Mimo utrudnień model wciąż był w stanie osiągnąć zadowalające wyniki, bliskie tym podanym przez autorów.

---

## 6. Źródła
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). **ArcFace: Additive Angular Margin
Loss for Deep Face Recognition.** CVPR 2019. [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
- [InsightFace](https://github.com/deepinsight/insightface)
- Ng, H.-W., & Winkler, S. (2014). **A data-driven approach to cleaning large face
datasets.** ICIP 2014. *(FaceScrub dataset)*
- [ChromaDB](https://www.trychroma.com/)
- [TinyFace](https://qmul-tinyface.github.io/)
- [Real-ESRGAN](https://github.com/xinntao/real-esrgan)
