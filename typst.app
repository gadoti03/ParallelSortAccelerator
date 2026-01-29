#import "@preview/diatypst:0.9.0": *

#show: slides.with(
  title: "Studio comparativo di algoritmi di ordinamento: SIMD e CUDA",
  subtitle: "Sistemi di Elaborazione Accelerata",
  date: "A.A. 2025/2026",
  authors: ("Gabriele Doti – 0001245897"),
  title-color: blue.darken(50%),
  ratio: 4/3,
  layout: "medium", // one of "small", "medium", "large"
  toc: true,
  count: "dot", // one of "dot", "dot-section", "number", or none
  footer: true,
  theme: "full",
  footer-title: "Studio comparativo di algoritmi di ordinamento",
  // footer-subtitle: "Custom Subtitle",
  // theme: "full", // one of "normal", "full"
  // ... see the docs for more options
)

= Introduzione

== Obiettivi e vincoli

Obiettivi del progetto e principali vincoli sui dati considerati.

*Obiettivi*:

- Analizzare implementazioni ottimizzate di algoritmi di ordinamento
- Vedere quanto le versioni ottimizzate (SIMD o CUDA) sono più veloci o efficienti rispetto alla versione seriale dell’algoritmo.
- Evidenziare scelte implementative chiave

*Vincoli*:
- Grandezza dell’array: multipla di 2
- Elementi: interi positivi (estendibile ad altri tipi)

= Algoritmi
== Algoritmi implementati
#table(
  columns: 4,
  align: (center, center, center, center),
  [Algoritmo], [Paradigma], [Complessità media], [Note],
  [Quick sort], [Divide & conquer], [O(n log n)], [Cache-friendly, \ non stabile],
  [Merge sort], [Divide & conquer], [O(n log n)], [Stabile, facilmente parallelizzabile],
  [Radix sort], [Non-comparison], [O(kn)], [Tempo proporzionale al numero di bit],
  [Bitonic sort], [Sorting network], [O(n log² n)], [Ottimo per parallelismo]
)
- Radix sort: ottimo per interi, scaling lineare con bit  
- Bitonic sort: adatto a parallelismo massivo

= Implementazione Seriale

== QuickSort

Implementazione di QuickSort:

```cpp
void quicksort_rec(unsigned int* arr, int left, int right) {
    if(left >= right) return;
    unsigned int pivot = arr[right];
    int i = left;
    for(int j = left; j < right; ++j) {
        if(arr[j] < pivot) {
            std::swap(arr[i], arr[j]);
            i++;
        }
    }
    std::swap(arr[i], arr[right]); // Place pivot
    quicksort_rec(arr, left, i-1);
    quicksort_rec(arr, i+1, right);
}

void quicksort(unsigned int* arr, int n) {
    quicksort_rec(arr, 0, n-1);
}
```
*Analisi Parallelismo:*

- Parallelizzabile solo sui sotto-array ricorsivi

- Dipende dai pivot → carico non bilanciato

- Overhead thread può annullare i guadagni

== MergeSort

Implementazione di QuickSort:

```cpp
void merge(unsigned int* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    // Create temporary arrays
    unsigned int* L = new unsigned int[n1];
    unsigned int* R = new unsigned int[n2];
    // Fill L and R arrays
    for(int i=0;i<n1;i++) L[i]=arr[left+i];
    for(int i=0;i<n2;i++) R[i]=arr[mid+1+i];
    // Merge the two halves
    int i=0,j=0,k=left;
    while(i<n1 && j<n2) arr[k++] = (L[i]<=R[j]) ? L[i++] : R[j++];
    while(i<n1) arr[k++]=L[i++];
    while(j<n2) arr[k++]=R[j++];
    // Deallocate temporary arrays
    delete[] L; delete[] R;
}
merge_sort(unsigned int* arr, int n) {
    if(n <= 1) return;
    int mid = n/2;
    merge_sort(arr, mid);
    merge_sort(arr+mid, n-mid);
    merge(arr, 0, mid-1, n-1); // arr, left, mid, right
}
```
*Analisi Parallelismo:*

- Ogni metà dell’array può essere ordinata in parallelo

- Fusione finale richiede sincronizzazione

- Scalabile meglio di QuickSort

== RadixSort

```cpp
...
unsigned int* output = new unsigned int[n];
unsigned int* count  = new unsigned int[bucket_size];

for (shift = 0; shift < max_bits; shift += bits_per_pass) {
    // Reset histogram
    std::memset(count, 0, bucket_size * sizeof(unsigned int));
    // Count occurrences of each value
    for (i = 0; i < n; i++) {
        unsigned int digit = (arr[i] >> shift) & mask;
        count[digit]++;
    }
    // Transform into cumulative position
    for (i = 1; i < bucket_size; i++)
        count[i] += count[i - 1];
    // Fill the output array in reverse order for stability
    for (i = n; i-- > 0; ) {
        unsigned int digit = (arr[i] >> shift) & mask;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }
    // Copy back to arr
    std::memcpy(arr, output, n * sizeof(unsigned int));
}
...
```

*Analisi Parallelismo:*

- Conteggio dei digit/bit può essere fatto in parallelo

- Riordinamento richiede sincronizzazione tra thread

- Performance dipende dal numero di bit per passaggio

== BitonicSort

```cpp
void bitonic_merge(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
    if(cnt>1){
        unsigned int k = cnt/2;
        for(unsigned int i=low;i<low+k;i++)
            if(dir==(arr[i]>arr[i+k]))
              std::swap(arr[i],arr[i+k]);
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low+k, k, dir);
    }
}
static void bitonic_sort_rec(unsigned int* arr, unsigned int low, unsigned int cnt, bool dir) {
    if(cnt>1){
        unsigned int k = cnt/2;
        // sort ascending
        bitonic_sort_rec(arr, low, k, true);
        // sort descending
        bitonic_sort_rec(arr, low+k, k, false);
        bitonic_merge(arr, low, cnt, dir);
    }
}

void bitonic_sort(unsigned int* arr, unsigned int n) {
    bitonic_sort_rec(arr, 0, n, true);
}
```

*Analisi Parallelismo:*

- I confronti possono essere eseguiti in parallelo, senza dipendenze


- Complessità seriale più alta, ma parallelismo massivo lo rende competitivo su hardware parallelo

#block(width: 100%)[
  #table(
    columns: (2fr, 1fr),
    [*Algorithm*], [*Time (ms)*],
    [Serial QuickSort], [3585.01],
    [Serial MergeSort], [3660.38],
    [Serial Radix (4 bit x part)], [496.935],
    [Serial Radix (8 bit x part)], [270.456],
    [Serial Radix (16 bit x part)], [236.828],
    [Serial Bitonic], [12779.3],
  )
]

    
= Implementazione SIMD (Intrinsics)

== Motivazione scelta algoritmi

- *QuickSort:*
  - Partizionamento dipendente dai pivot -> dati irregolari, difficile vettorizzare e creare shuffle su più registri

- *MergeSort:*
  - Fusione dei dati richiede di combinare elementi provenienti da più registri SIMD, rendendo lo shuffle complesso e poco efficiente  

- *RadixSort:*
  - Operazioni principali: conteggio e spostamento dei bit, possono essere eseguite simultanemente in vettori
  
- *BitonicSort:*
  - Alcuni confronti su un livello possono essere eseguiti simultaneamente in vettori
  
== RadixSort

*Definizioni preliminari*:

- *Nibble*: sequenza di *4 bit*, quindi valori da 0 a 15.

```cpp
unsigned int bits_per_pass = 4;

unsigned int histogram_size = std::pow(2, bits_per_pass);
```

- *Istogramma*: struttura dati che conta quante volte compare ogni valore possibile di un certo intervallo.

```cpp
unsigned int* count = static_cast<unsigned int*>(
    _mm_malloc(histogram_size * sizeof(unsigned int), 32)
);
```


```cpp
...
for (i = 0; i < n; i+=integers_per_reg * 4) {
    // Zero the digits accumulator
    digits_accumulator = _mm256_setzero_si256();
    for (j = 0; j < 4; j++) {
        // Read 8 integers
        vec = _mm256_load_si256((__m256i*)&arr[i+j*integers_per_reg]);
        // Extract the relevant 4 bits
        digits = _mm256_and_si256(_mm256_srli_epi32(vec, shift), _mm256_set1_epi32(mask));
        // Shift digits to their position in the accumulator
        digits = _mm256_slli_epi32(digits, j * bits_per_pass * (8 / bits_per_pass));
// Accumulate digits
        digits_accumulator = _mm256_or_si256(digits_accumulator, digits);
    }
    
    // Update count array
    for (index = 0; index < histogram_size; index++) {
        // Compare digits with index
        cmp = _mm256_cmpeq_epi8(digits_accumulator, _mm256_set1_epi8(index));
        // Take the most significant bit of each byte to form a mask (integer with 32 bits
        mask_cmp = _mm256_movemask_epi8(cmp); // Compact comparison results into a 32-bit integer
        // Count number25 of set bits in mask_cmp
        count_set_bits = __builtin_popcount(mask_cmp);
        count[index] += count   _set_bits;
    }
}
...
```
*Operazioni di Shift*

- Eseguo 4 letture di registri a 256bit
```cpp
for (i = 0; i < n; i+=integers_per_reg * 4) {...}
```
- Leggo 256 bit
```cpp
    _m256i vec = _mm256_load_si256((__m256i*)&arr[i+j*8]);
```
#image("1.png")
\
\
\
\
- Estraggo i 4 bit di interesse (nibble)
```cpp
    _m256i digits = _mm256_and_si256(_mm256_srli_epi32(vec, shift), _mm256_set1_epi32(mask));
```
#image("2.png")
\
\
\
\
\
\
\
\
\
- Shifto ogni nibble nella corretta posizione
```cpp
    _m256i digits = _mm256_slli_epi32(digits, j * bits_per_pass * (8 / bits_per_pass));
```
#image("3.png")
- Inserisco ogni nibble nell'accumulatore, nella corretta posizione
```cpp
    _m256i digits_accumulator = _mm256_or_si256(digits_accumulator, digits);
```
*Operazioni di Conteggio*

- Per ogni posizione nell'istogramma
```cpp
for (index = 0; index < histogram_size; index++) {
```
- Comparo l'index con ognuno degli ultimi 32 valori letti (ottengo maschera)
```cpp
    _m256i cmp = _mm256_cmpeq_epi8(digits_accumulator, _mm256_set1_epi8(index));
```
#image("4.png")
- Compatto la maschera ottentua in un intero a 32 bit
```cpp
    int mask_cmp = _mm256_movemask_epi8(cmp)
```
- Conto il numero dei bit settati all'interno dell'intero mask_cmp
```cpp
    unsigned int count_set_bits = __builtin_popcount(mask_cmp);
    ...
```

#block(width: 100%)[
  #table(
    columns: (1fr, 1fr, 1fr),
    [*Versione*], [*Tempo*], [*Speedup*],
    [Seriale (4 bit x part)], [496.935], [-],
    [SIMD (intrinsics)], [234.815], [2.11×],
  )
]

== BitonicSort
*Fasi dell'ordinamento*:
1. Confronto a distanza fissa _k_
  - Gli elementi vengono confrontati a coppie con distanza k
  - Operazioni regolari e indipendenti -> SIMD facilmente applicabile
2. Ordinamento delle sottosequenze
  - Ogni metà viene ordinata in modo crescente o decrescente
```cpp
  if (k >= integers_per_reg) {
      unsigned int i = low;
      unsigned int simd_end = low + k;
      for (; i < simd_end; i += integers_per_reg) {
          // load first half
          __m256i a = _mm256_load_si256((__m256i*)&arr[i]);
          // load second half
          __m256i b = _mm256_load_si256((__m256i*)&arr[i + k]);

          __m256i lo = _mm256_min_epu32(a, b);
          __m256i hi = _mm256_max_epu32(a, b);

          if (!dir) std::swap(lo, hi);
          
          _mm256_store_si256((__m256i*)(arr + i), lo);
          _mm256_store_si256((__m256i*)(arr + i + k), hi);
      }
  }
```
#block(width: 100%)[
  #table(
    columns: (1fr, 1fr, 1fr),
    [*Versione*], [*Tempo*], [*Speedup*],
    [Seriale], [12277.2], [-],
    [SIMD (intrinsics)], [3361.45], [3.65×],
  )
]



= 
= 













































= Pesce tosto
== U pesce 

Then, insert your content.

- Level-one headings corresponds to new sections.
- Level-two headings corresponds to new slides.

== Options

To start a presentation, only the title key is needed, all else is optional!

Basic Content Options:
#table(
  columns: 3,
  [*Keyword*], [*Description*], [*Default*],
  [_title_], [Title of your Presentation, visible also in footer], [`none` (required!)],
  [_subtitle_], [Subtitle, also visible in footer], [`none`],
  [_date_], [a normal string presenting your date], [`none`],
  [_authors_], [either string or array of strings], [`none`],
  [_footer-title_], [custom text in the footer title (left)], [same as `title`],
  [_footer-subtitle_], [custom text in the footer subtitle (right)], [same as `subtitle`],
)

#pagebreak()

Advanced Styling Options:
#table(
  columns: 3,
  [*Keyword*], [*Description*], [*Default*],
  [_layout_], [one of _small, medium, large_, adjusts sizing of the elements on the slides], [`"medium"`],
  [_ratio_], [aspect ratio of the slides, e.g., 16/9], [`4/3`],
  [_title-color_], [Color to base the Elements of the Presentation on], [`blue.darken(50%)`],
  [_bg-color_], [Background color of the slides, can be any color], [`white`],
  [_count_], [one of _dot, number, none_, adjusts the style of page counter in the right corner], [`"dot"`],
  [_footer_], [whether to display the footer at the bottom], [`true`],
  [_toc_], [whether to display the table of contents], [`true`],
  [_theme_], [one of _normal, full_, adjusts the theme of the slide], [`"normal"`],
  [_first-slide_], [whether to include the default title slide], [`true`],
)

The full theme adds more styling to the slides, similar to a a full LaTeX beamer theme.

= Default Styling in diatypst

== Terms, Code, Lists

_diatypst_ defines some default styling for elements, e.g Terms created with ```typc / Term: Definition``` will look like this

/ *Term*: Definition

A code block like this

```python
// Example Code
print("Hello World!")
```


