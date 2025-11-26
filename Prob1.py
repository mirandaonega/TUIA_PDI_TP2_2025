# Problema 1 - Fácil
	# Monedas: se reconocen por el area
	# Falta que demos algo de teoría la semana que viene para hacerlo
	# Dados: reconocer con bounding box y decir el número al que corresponde SOLO la cara superior

import cv2
import numpy as np
import matplotlib.pyplot as plt

monedas = cv2.imread('monedas.jpg')   # Leemos imagen

def detectar_letras(ruta_imagen):
    imagen = ruta_imagen
    imagen_con_bordes = imagen.copy()
    
    if imagen is None:
        print("Error: No se pudo cargar la imagen. Verifique la ruta.")
        return

    # 1. Preprocesamiento: Convertir a gris y desenfoque (opcional, pero ayuda)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris_blur = cv2.medianBlur(gris, 3) # Elimina ruido sin erosionar el texto

    # --- PASO CLAVE: UMBRALIZACIÓN (Binarización) ---
    # Convertimos la imagen a blanco y negro puro. 
    # Usamos THRESH_BINARY_INV para que el texto sea blanco (255) y el fondo negro (0),
    # que es lo que 'findContours' espera para un buen resultado.
    # El valor 180 es un umbral de ejemplo; ajústalo si tu texto no es puramente blanco.
    _, binaria_invertida = cv2.threshold(gris_blur, 180, 255, cv2.THRESH_BINARY_INV)

    # Nota: Si el texto está dividido, usa cv2.ADAPTIVE_THRESH_GAUSSIAN_C en lugar de cv2.threshold.
    # binaria_invertida = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. ENCONTRAR CONTORNOS en la imagen binaria (saltando Canny y morfología)
    # Los contornos ahora rodearán cada "mancha" blanca (cada letra)
    contornos, _ = cv2.findContours(binaria_invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Ajuste del Filtrado para Letras Individuales
    min_ancho = 2 
    min_alto = 2  
    max_ancho = 30 
    max_alto = 40  
    
    #print("Contornos detectados (letras):")
    
    cant_caracteres=0

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Filtrar por tamaño de letra
        if w > min_ancho and h > min_alto and w < max_ancho and h < max_alto: 
            # Dibujar el rectángulo (verde)
            cv2.rectangle(imagen_con_bordes, (x, y), (x + w, y + h), (0, 255, 0), 1) 
            cant_caracteres+=1
            
    # 4. Mostrar resultado
    #rgb_imagen = cv2.cvtColor(imagen_con_bordes, cv2.COLOR_BGR2RGB)
    
    #plt.figure(figsize=(10, 8))
    #plt.imshow(rgb_imagen)
    #plt.title("Detección de Letras Individuales (Binarización)")
    #plt.axis('off')
    #plt.show()
    return cant_caracteres
