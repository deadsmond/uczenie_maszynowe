def dzialanie1(A, x):
    """ iloczyn macierzy A z wektorem x """
    return A.dot(x) # tu wpisz odpowiedź

def dzialanie2(A, B):
    """ iloczyn macierzy A · B """
    return A.dot(B) # tu wpisz odpowiedź

def dzialanie3(A):
    """ odwrotność iloczynu A · A.T """
    return np.linalg.inv(A.dot(np.transpose(A))) # tu wpisz odpowiedź

def dzialanie4(A, B):
    """ wynik działania (A · B)^T - B^T · A^T """
    one = np.transpose(A.dot(B))
    two = np.transpose(B)
    three = np.transpose(A)
    return one - two.dot(three) # tu wpisz odpowiedź
