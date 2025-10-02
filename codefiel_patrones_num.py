import random



# ========= Utilidades =========
def print_formateado(vector, filas=8, columnas=8):
    for i in range(filas):
        print(''.join('█' if v == 1 else ' ' for v in vector[i*columnas:(i+1)*columnas]))
    print()


def leer_matriz(nombre_archivo):
    matriz = []
    with open(nombre_archivo, "r") as f:
        for linea in f:
            
            fila = [int(x) for x in linea.strip().split()]
            matriz.append(fila)
    return matriz

# ========= Patrones (num1..num5) =========
num1 =  leer_matriz("Dataset/num1.txt")
num2 =  leer_matriz("Dataset/num2.txt")
num3 =  leer_matriz("Dataset/num3.txt")
num4 =  leer_matriz("Dataset/num4.txt")
num5 =  leer_matriz("Dataset/num5.txt")

patrones = [num1, num2, num3, num4, num5]
patrones_flat = [[item for fila in p for item in fila] for p in patrones]

# ========= Construcción de la matriz de pesos =========
def trasnpuesta(a):
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]

def multiplicar_matriz(matriz, matrizb):
    filas = len(matriz)
    columnas = len(matrizb[0])
    out = [[0 for _ in range(columnas)] for _ in range(filas)]
    for i in range(filas):
        for j in range(columnas):
            out[i][j] = sum(matriz[i][k] * matrizb[k][j] for k in range(len(matrizb)))
    return out

def suma_matrices_lista(lista_matrices):
    filas, columnas = len(lista_matrices[0]), len(lista_matrices[0][0])
    out = [[0 for _ in range(columnas)] for _ in range(filas)]
    for matriz in lista_matrices:
        for i in range(filas):
            for j in range(columnas):
                out[i][j] += matriz[i][j]
    return out

def diagonal_cero(matriz):
    for i in range(len(matriz)):
        matriz[i][i] = 0

lista_w = []
for p in patrones_flat:
    vector_columna = [[v] for v in p]
    w = multiplicar_matriz(vector_columna, trasnpuesta(vector_columna))
    lista_w.append(w)

W = suma_matrices_lista(lista_w)
diagonal_cero(W)

# ========= Hopfield Recall con ruido inicial =========
def hopfield_recall(W, patron_inicial, max_iter=300, filas=8, columnas=8, patrones_guardados=None, ruido=0.05):
    # --- agregar ruido a la entrada ---
    estado = patron_inicial[:]
    N = len(estado)
    entrada_original = patron_inicial[:]
    for i in range(N):
        if random.random() < ruido:
            estado[i] *= -1  # invertir aleatoriamente algunos bits

    mejor_estado = estado[:]

    print("Patrón inicial ")
    print_formateado(patron_inicial, filas, columnas)

    for it in range(max_iter):
        cambios = 0

        # Actualización sincrónica
        nuevo_estado = []
        for i in range(N):
            val = sum(W[i][j] * estado[j] for j in range(N))
            nuevo_val = 1 if val > 0 else (-1 if val < 0 else estado[i])
            nuevo_estado.append(nuevo_val)
            if nuevo_val != estado[i]:
                cambios += 1

        estado = nuevo_estado[:]

        # Guardar estado más parecido a la entrada original
        distancia = sum(1 for x, y in zip(estado, entrada_original) if x != y)
        if distancia < sum(1 for x, y in zip(mejor_estado, entrada_original) if x != y):
            mejor_estado = estado[:]

        # Detener si coincide con algún patrón conocido
        if patrones_guardados and estado in patrones_guardados:
            print(f"Patrón exacto conocido encontrado en iteración {it}")
            print_formateado(estado, filas, columnas)
            return estado

        # Imprimir cada 20 iteraciones
        if it % 20 == 0 or it == max_iter-1:
            print(f"Iteración {it}")
            print_formateado(estado, filas, columnas)

        # Si no hubo cambios → la red se estabilizó
        if cambios == 0:
            print(f"Red estabilizada en iteración {it}")
            break

    print("Patrón más parecido a la entrada encontrado:")
    print_formateado(mejor_estado, filas, columnas)
    return mejor_estado

# ========= Patrón de prueba =========
entrada_prueba = leer_matriz("Dataset/valor_entrada.txt")
entrada_flat = [item for fila in entrada_prueba for item in fila]

# ========= Ejecutar =========
resultado = hopfield_recall(W, entrada_flat, patrones_guardados=patrones_flat, ruido=0.05)
