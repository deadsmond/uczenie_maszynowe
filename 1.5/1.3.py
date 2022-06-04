def wlasciwosci_macierzy(A):
    num_rows, num_cols = A.shape
    liczba_elementow = num_cols * num_rows # tu uzupełnij
    liczba_kolumn = num_cols # tu uzupełnij
    liczba_wierszy = num_rows # tu uzupełnij
    srednie_wg_wierszy = A.mean(1) # tu uzupełnij
    srednie_wg_kolumn = A.mean(0) # tu uzupełnij
    kolumna_2 = A.T[2] # tu uzupełnij
    wiersz_3 = A[3] # tu uzupełnij
    return (
        liczba_elementow, liczba_kolumn, liczba_wierszy,
        srednie_wg_wierszy, srednie_wg_kolumn,
        kolumna_2, wiersz_3)
