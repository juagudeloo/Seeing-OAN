"""________________Distance_IMG______________
Programa que construye el histograma de frecuencias de la separación 
entre dos "spots" (objetos) en el DIMM."""

"""_________________Libraries________________________"""
import cv2
import glob
import numpy as np
from scipy.stats import norm
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import re
"""__________________________________________________"""



"""__________________FUNCTIONS_______________________"""
# Definir una función con parámetros de Open CV
def identify_stars_and_distance(image_path):
    # Cargar imagen con filtro en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    # Aplicar un umbral para binarizar la imagen y destacar los objetos (estrellas) Este parámetro se puede variar según la intensidad de la estrella
    _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)


    # Encontrar los contornos de los objetos (estrellas)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Filtrar unicamente los dos contornos más grandes detectados (asumiendo que son las estrellas)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    # Obtener los centros de las dos estrellas
    star_centers = []  #crear un vector para guardar los centros
    for contour in contours:
        M = cv2.moments(contour) #calcula los momentos del contorno, función de Open CV
        if M["m00"] != 0:  # moo es el area del contorno
            cX = int(M["m10"] / M["m00"])  #m10 es el momento espacial para calcular las coordenadas del centroide
            cY = int(M["m01"] / M["m00"])  #m01 es el momento espacial para calcular las coordenadas del centroide
            star_centers.append((cX, cY)) #guardar las coordenadas de los centros encontrados


    # Calcular la distancia entre las dos estrellas
    if len(star_centers) == 2:
        pixel_distance = dist.euclidean(star_centers[0], star_centers[1])
    else:
        print("No se encontraron estrellas.")

    # Mostrar la imagen con los centros de las estrellas marcados
    #plt.imshow(image, cmap='gray')
    #plt.title(f"Distancia en pixeles: {pixel_distance:.2f}")
    #plt.show()
    

    return(pixel_distance)

# Definir una funcion para extraer el numero del frame dek nombre del archivo
def extract_frame_number(filename):
    match = re.search(r'frame(\d+)\.jpg', filename)
    return int(match.group(1)) if match else -1

"""__________________________________________________"""



"""_____________________SCALE________________________"""
scale_size = 1.0  # segundos de arco por px ('' /px)

"""__________________________________________________"""



"""_____________________MAIN_________________________"""
# Abrir carpeta donde se encuentran los frames del video
carpeta = r"D:\GitHub\Tecnicas\Seeing\Scripts\data\G-1_vega_270824_21_09_DSC_0019_mascara"


# Busqueda de los archivos .jpg
archivos = glob.glob(carpeta + r"\*.jpg")

# Ordenar los archivos numéricamente por el número de frame en el nombre asi (0, 1, 2, 3, ...)
archivos.sort(key=extract_frame_number)

# Definir los frames que se procesaran, se escogieron unicamente los frames entre el 40% y el 80% para eliminar errores al inicio y fin
start = int(len(archivos)*0.40)
end = int(len(archivos)*0.80)

nombres = []
for j in archivos:
    if carpeta in j:
        nombres.append(j.replace(carpeta, ""))


if nombres != []:
    l = len(nombres)
    print(f"\n Su carpeta tiene {l} archivos .jpg \n")
    print(f"          ....            ")
    print(f"\nSe procesarán los frames del:{start} al:{end}\n")
    print(f"          ....            ")
    print(f"No. 1: {nombres[start]}")
    print(f"          ....            ")
    print(f"No. {end}: {nombres[end]}")
    print(f"          ....            ")
    print(f"          ....            ")
    print("\n :: CONSTRUYENDO EL HISTOGRAMA :: \n")
    print(f"          ....            ")
else:
    print("\n Su carpeta no tiene archivos .jpg \n")

distances = []


for k in range(start, end):
    # Usar la función en un frame
    identify_stars_and_distance(archivos[k])
    #print(f"archivo {archivos[k]}") #verificar si se están ejecutando los frames en orden
    distances.append(identify_stars_and_distance(archivos[k])*scale_size)


# Guardar las distancias en un archivo llamado out.dat
np.savetxt("out.dat", distances, fmt="%.6f", header="Distancias entre estrellas (arcsec)")


# Ajuste de los datos a una distribución normal
mu, std = norm.fit(distances)
# Normalizar los valores del histograma
weights = np.ones_like(distances) / len(distances)
# Graficar el histograma con las distancias
plt.hist(distances, weights=weights, bins=10, alpha=0.6, color="g", edgecolor="black")
# Graficar la función de densidad de probabilidad (PDF) ajustada
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, "k", linewidth=2)


# Mostrar los valores de mu y sigma
print(f"Media (mu): {mu:.3f}, Desviación estándar (sigma): {std:.3f}")


# Grafico
plt.title(r"$\mathrm{Histograma\ de\ distancias\ entre\ estrellas:}\ \mu=%.3f,\ \sigma=%.3f$" % (mu, std))
plt.xlabel("Distancia entre los centroides (arcsec)")
plt.ylabel("Frecuencia")
plt.text(0.15, 0.9, f"Placa: {scale_size} \"/px", transform=plt.gca().transAxes, fontsize=12, color='black', ha='center')
plt.show()