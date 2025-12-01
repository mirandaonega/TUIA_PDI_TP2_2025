#Detectar bordes
import cv2
import numpy as np
import matplotlib.pyplot as plt

def dividir_pares_ordenados(lista):
    if len(lista) < 2:
        return []
    lista_ordenada = sorted(lista)

    # Dividir pares consecutivos (siguiente / actual)
    ratios = []
    for i in range(len(lista_ordenada) - 1):
        ratio = lista_ordenada[i + 1] / lista_ordenada[i]
        ratios.append(ratio)

    return ratios

def contar_huecos(img_original, imagen_binaria, posicion):
    """La función recibe la imagen original, una binarizada y la posición del
    contorno a procesar, luego se extrae la ROI y se la procesa para poder
    detectar los contorno interntos."""

    #Detectar contornos externos
    contornos_ext, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Encuentra las coordenadas de los dados
    x, y, ancho, alto = cv2.boundingRect(contornos_ext[posicion])

    #Extrae la ROI usando las coordenadas del dado
    roi = img_original[y:y+alto, x:x+ancho]

    #Aplicar borrosidad
    gau = cv2.GaussianBlur(roi,(21,21),0)

    #Detectar bordes
    can = cv2.Canny(gau,80,205)

    s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    clausura_img = cv2.morphologyEx(can, cv2.MORPH_CLOSE, s, iterations=3)

    #Encontrar contornos internos
    contornos, jerarquia = cv2.findContours(clausura_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Contar los contornos encontrados (huecos = contornos con padre)
    contornos_internos = sum(1 for h in jerarquia[0] if h[3] != -1)

    #Dibujar contornos
    roi_contorno = roi.copy()
    for i, contour in enumerate(contornos):
        if jerarquia[0][i][3] != -1:
            dibujar_contorno = cv2.drawContours(roi_contorno, [contour], -1, (255, 0, 0), thickness=15)

    return contornos_internos, roi, gau, can, dibujar_contorno

def procesar_monedas_y_dados(img):
    """
    Función principal que procesa una imagen para detectar y clasificar monedas y dados.
    """
    imagen_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pasar a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar borrosidad
    gauss = cv2.GaussianBlur(img_gray,(11,11),0)

    # Detectar bordes
    canny = cv2.Canny(gauss,7,75)

    # Aplicar dilatación
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    img_dilatacion = cv2.dilate(canny,kernel,iterations=3)

    # Aplicar erosión
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    img_erosion = cv2.erode(img_dilatacion,kernel,iterations=2)

    # Morfología para mejorar la segmentación obtenida
    seg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    # Clausura para rellenar huecos.
    clausura_img = cv2.morphologyEx(img_erosion, cv2.MORPH_CLOSE, seg, iterations=3)

    # Mostrar la imagen original y los resultados
    plt.figure(figsize=(15,8),facecolor='#333333')
    plt.subplot(231), plt.imshow(imagen_original), plt.title('Imagen Original',color='white'),plt.axis('off')
    plt.subplot(232), plt.imshow(gauss, cmap='gray'), plt.title('Imagen con borrosidad (Gauss)',color='white'),plt.axis('off')
    plt.subplot(233), plt.imshow(canny, cmap='gray'), plt.title('Detección de lineas (Canny)',color='white'),plt.axis('off')
    plt.subplot(234), plt.imshow(img_dilatacion, cmap='gray'), plt.title('Imagen dilatada',color='white'),plt.axis('off')
    plt.subplot(235), plt.imshow(img_erosion, cmap='gray'), plt.title('Imagen erosionada',color='white'),plt.axis('off')
    plt.subplot(236), plt.imshow(clausura_img, cmap='gray'), plt.title('Imagen con clausura aplicada',color='white'),plt.axis('off')
    plt.subplots_adjust(hspace=0.1)
    plt.show()

    contornos, _ = cv2.findContours(clausura_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    monedas = []
    dados = []
    dados_puntos = []
    dados_visualizacion = []  # Para guardar las imágenes del proceso
    monedas_areas = []
    monedas_indices = []
    umbral_de_area = 1500
    umbral_circularidad = 0.8
    nombres_monedas = ['10c', '1p', '50c']

    for i, cnt in enumerate(contornos):
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)

        if perimetro > 0 and area > umbral_de_area:  # Filtrar objetos muy pequeños
            # Calcular circularidad: 4pi*área
            # Círculo perfecto = 1
            circularidad = 4 * np.pi * area / (perimetro ** 2)

            if circularidad > umbral_circularidad:  # Umbral para círculos
                monedas.append(cnt)
                monedas_areas.append(area)
                monedas_indices.append(i)
            else:
                dados.append(cnt)
                nro_dados, roi, gau, can, dib_cont = contar_huecos(imagen_original, clausura_img, i)
                dados_puntos.append(nro_dados)
                dados_visualizacion.append((roi, gau, can, dib_cont, nro_dados))

    # Visualizar monedas y dados
    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_result, monedas, -1, (0, 255, 0), 2)  # Verde
    cv2.drawContours(img_result, dados, -1, (255, 0, 0), 2)    # Rojo

    plt.figure(figsize=(10, 6))
    plt.imshow(img_result)
    plt.title(f'Monedas: {len(monedas)} (verde) | Dados: {len(dados)} (rojo)')
    plt.axis('off')
    plt.show()

    # Mostrar el proceso de detección de puntos para cada dado en 2 filas
    num_dados = len(dados_visualizacion)
    if num_dados > 0:
        plt.figure(figsize=(15, 8), facecolor='#333333')

        for idx, (roi, gau, can, dib_cont, puntos) in enumerate(dados_visualizacion):
            # Fila 1: Dado 1, Fila 2: Dado 2
            base = idx * 4  # 0 para primer dado, 4 para segundo dado

            plt.subplot(2, 4, base + 1), plt.imshow(roi), plt.title(f'ROI Dado {idx+1} - {puntos} puntos', color='white'), plt.axis('off')
            plt.subplot(2, 4, base + 2), plt.imshow(gau, cmap='gray'), plt.title('Borrosidad (Gaussian)', color='white'), plt.axis('off')
            plt.subplot(2, 4, base + 3), plt.imshow(can, cmap='gray'), plt.title('Detección de bordes (Canny)', color='white'), plt.axis('off')
            plt.subplot(2, 4, base + 4), plt.imshow(dib_cont), plt.title('Contornos detectados', color='white'), plt.axis('off')

        plt.tight_layout()
        plt.show()

    ratios = dividir_pares_ordenados(monedas_areas)

    # Identificar saltos grandes (> 0.1) para separar tipos de monedas
    areas_ordenadas = sorted(monedas_areas)
    saltos = []
    for i, ratio in enumerate(ratios):
        if ratio > 1.1:  # Salto mayor a 10%
            saltos.append(i)
            print(f"Salto detectado en índice {i}")

    # Crear grupos basados en los saltos
    grupos_areas = []
    inicio = 0
    for salto in saltos:
        grupos_areas.append(areas_ordenadas[inicio:salto+1])
        inicio = salto + 1
    # Agregar el último grupo
    grupos_areas.append(areas_ordenadas[inicio:])

    # Asignar nombres a los grupos (10c, 50c, 1p) - de menor a mayor área
    print(f"\nGrupos detectados: {len(grupos_areas)}")
    for i, grupo in enumerate(grupos_areas):
        nombre = nombres_monedas[i] if i < len(nombres_monedas) else f'Tipo {i}'
        print(f"  {nombre}: {len(grupo)} monedas")

    # Crear mapeo de área a tipo de moneda
    area_a_tipo = {}
    for tipo_idx, grupo in enumerate(grupos_areas):
        for area in grupo:
            area_a_tipo[area] = tipo_idx

    # Crear imagen con marcadores
    img_marcada = imagen_original.copy()

    # Colores para cada tipo (RGB)
    colores = [
        (255, 0, 0),    # Rojo - 10c
        (0, 255, 0),    # Verde - 50c
        (0, 0, 255),    # Azul - 1p
    ]

    # Dibujar contornos y etiquetar centroides
    for contorno, area in zip(monedas, monedas_areas):
        tipo = area_a_tipo[area]
        color = colores[tipo % len(colores)]
        nombre = nombres_monedas[tipo] if tipo < len(nombres_monedas) else f'T{tipo}'

        # Dibujar contorno
        cv2.drawContours(img_marcada, [contorno], -1, color, 2)

        # Calcular centro usando bounding box
        x, y, w, h = cv2.boundingRect(contorno)
        cx = x + w // 2
        cy = y + h // 2

        # Escribir etiqueta
        cv2.putText(img_marcada, nombre, (cx - 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Dibujar dados en blanco
    cv2.drawContours(img_marcada, dados, -1, (255, 255, 255), 2)

    # Agregar número de puntos en cada dado (parte superior izquierda)
    for dado, puntos in zip(dados, dados_puntos):
        x, y, _, _ = cv2.boundingRect(dado)
        # Posición superior izquierda del dado
        text_x = x + 10
        text_y = y + 60

        cv2.putText(img_marcada, str(puntos), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 12)

        cv2.putText(img_marcada, str(puntos), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 6)

    altura_img, _ = img_marcada.shape[:2]
    y_inicio = altura_img - 480
    x_inicio = 20

    # Contar monedas por tipo
    contador_tipos = {0: 0, 1: 0, 2: 0}
    for i, grupo in enumerate(grupos_areas):
        contador_tipos[i] = len(grupo)

    cv2.putText(img_marcada, f"10c: {contador_tipos[0]} monedas", (x_inicio, y_inicio + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 0, 0), 8)
    cv2.putText(img_marcada, f"50c: {contador_tipos[1]} monedas", (x_inicio, y_inicio + 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 255, 0), 8)
    cv2.putText(img_marcada, f"1p:  {contador_tipos[2]} monedas", (x_inicio, y_inicio + 320),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 255), 8)

    # Mostrar resultado
    plt.figure(figsize=(12, 8))
    plt.imshow(img_marcada)
    plt.title('Monedas clasificadas: Rojo=10c, Azul=50c, Verde=1p, Blanco=Dados')
    plt.axis('off')
    plt.show()

# Ejecutar la función con la imagen
path='' # En mi caso tengo que ponerle el path absoluto, pero lo dejo vacío como default
img = cv2.imread(path+'monedas.jpg')
procesar_monedas_y_dados(img)
