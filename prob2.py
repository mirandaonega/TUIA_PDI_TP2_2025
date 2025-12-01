import cv2
import numpy as np
import matplotlib.pyplot as plt

def obtener_patentes(imagen):
  """
  Aplica transformaciones a la imagen de entrada,
  y devuelve la subimagen de la patente
  """
  # Lectura y cambio de color
  img = cv2.imread(imagen, cv2.IMREAD_COLOR)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Filtro Sobel
  imagen_sobel = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=1)
  imagen_sobel = cv2.convertScaleAbs(imagen_sobel)

  # Binarización
  imagen_binaria = np.where(imagen_sobel > 108, 1, 0)
  imagen_binaria = imagen_binaria.astype(np.uint8)

  # Dilatación
  kernel = np.ones((16,16),np.uint8)
  img_dilatada = cv2.dilate(imagen_binaria,kernel,iterations=1)

  # Mejora de la segmentación obtenida
  morf = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))

  # Apertura
  imagen_apertura = cv2.morphologyEx(img_dilatada, cv2.MORPH_OPEN, morf)

  # Detección de componentes conectadas
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_apertura,8, cv2.CV_32S)
  comp = img.copy()
  for st in stats[1:]:
    bounding = cv2.rectangle(comp,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(255,0,0),thickness=10)

  # Guarda la mayor relación de aspecto
  relacion_aspecto_mayor = 0
  # Guarda las coordenadas de la img con mayor relación de aspecto (patente)
  xx,yy,ll,hh = 0,0,0,0

  for st in stats:
    # Filtra los bounding box por área (obtenida experimentalmente)
    if st[4] >= 1400 and st[4] <= 7000:

      # Extrae las coordenadas de las patentes
      x, y, ancho, alto = st[0],st[1],st[2],st[3]

      # Calcula relación de aspecto (mayor relación de aspecto -> fig más planas)
      relacion_de_aspecto = float(ancho) / alto

      # Reemplaza la relación de aspecto si se encuentra una mejor (mayor)
      if relacion_de_aspecto > relacion_aspecto_mayor:
        relacion_aspecto_mayor = relacion_de_aspecto
        xx,yy,ll,hh = x, y, ancho, alto

  # Extrae ROI usando las coordenadas
  roi = img[yy:yy+hh, xx:xx+ll]

  # los siguientes valores fueron calculados a mano
  umbral = 110
  numero = imagen[3:5]

  if numero in ('03','10'):
    umbral = 127
    img_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      # Si el píxel es > umbral, asigna 0 (negro); si no, asigna 255 (blanco)
    imagen_transformada = np.where(img_gris > umbral, 0, 255).astype(np.uint8)

  elif numero in ('04', '05','11'):
    umbral = 135
    img_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    imagen_transformada = np.where(img_gris > umbral, 0, 255).astype(np.uint8)

  elif numero in ('01', '08'):
    umbral = 145
    img_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    imagen_transformada = np.where(img_gris > umbral, 0, 255).astype(np.uint8)

  else:
    img_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    imagen_transformada = np.where(img_gris > umbral, 0, 255).astype(np.uint8)

  roi_binaria = cv2.bitwise_not(imagen_transformada) # invierte los colores

  return img, imagen_sobel, img_dilatada, imagen_apertura, bounding, roi, roi_binaria

def segmentar_patentes(img, img_binaria):
    """
    Recibe la imagen y la binarizacion de una patente y
    retorna la segmentación de los caracteres ordenados de izquierda a derecha.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binaria, 8, cv2.CV_32S)
    im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)

    rois_con_x = []
    patente_copia = img.copy()
    bounding = patente_copia.copy()

    for st in stats:
        x, y = st[0], st[1]
        w = st[2]
        h = st[3]
        area = st[4]

        # Ignora el fondo (label 0)
        if x == 0 and y == 0:
            continue

        aspect_ratio = w / (h + 1e-6)
        criterio_altura = h >= 10
        criterio_aspecto = 0.2 <= aspect_ratio <= 1.0

        if criterio_altura and criterio_aspecto:
            roi = img_binaria[y:y+h, x:x+w]
            rois_con_x.append((x, roi))   # Guarda x junto con la ROI

            # Dibujar bounding box
            cv2.rectangle(bounding, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Ordena por la coordenada X
    rois_con_x.sort(key=lambda t: t[0])
    rois_ordenadas = [roi for (_, roi) in rois_con_x]

    return img_binaria, bounding, rois_ordenadas, img

imgs = ['img01.png', 'img02.png', 'img03.png', 'img04.png', 'img05.png', 'img06.png', 'img07.png', 'img08.png', 'img09.png', 'img10.png', 'img11.png', 'img12.png']

for i in imgs:
    imag = obtener_patentes(i)
    patentes = segmentar_patentes(imag[5], imag[6])
    rois = patentes[2]

    # Se excluye patentes[3] que es el img original de la patente, ya está en imag[5])
    imagenes = [imag[0], imag[1], imag[2], imag[3], imag[4], imag[5], patentes[0], patentes[1], rois[0], rois[1], rois[2], rois[3], rois[4], rois[5]]

    # Reduce la cuadrícula a un tamaño adecuado para 14 imágenes
    fig, axes = plt.subplots(2, 7, figsize=(15, 6))
    axes = axes.flatten()
    titulos = ['Original','Sobel','Dilatación','Apertura','Bounding','ROI','ROI binaria','Bounding','char 1','char 2','char 3','num 1','num 2','num 3']
    idx = 0

    # Itera solo sobre las imágenes que caben en la cuadrícula
    for im in imagenes:
        if idx < len(axes):
            # Comprobación de dimensiones para el 'cmap' (opcional, pero buena práctica)
            if im.ndim == 3: # Imagen a color (H, W, 3)
                axes[idx].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) # Convierte BGR a RGB para Matplotlib
            else: # Imagen en escala de grises (H, W)
                axes[idx].imshow(im, cmap='gray')
            axes[idx].set_title(titulos[idx])
            axes[idx].axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()