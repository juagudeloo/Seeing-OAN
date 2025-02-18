"""
2. Frames analysis

Una vez se tienen los frames de los videos con los dos puntos (spots) en cada frame, se procede a analizar cada frame 
para determinar la distancia y variación de los puntos entra cada frame, obteniendo el movimiento aparente y relativo 
entre ellos.
"""
# Parámetros de observación del metodo DIMM

Dhole = 42  # diametro de la apertura  en mm
dsep = 144  # diametro de la separacion de las aperturas en mm
lamb = 0.0005  # longitud de onda en micrometros (mm)

"""________________Distance_IMG______________
Programa que construye el histograma de frecuencias de la separación 
entre dos "spots" (objetos) en el DIMM."""

"""_________________Libraries________________________"""
import cv2
import os
import glob
import numpy as np
from scipy.stats import norm
from scipy.spatial import distance as dist
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
from tqdm import tqdm

"""__________________________________________________"""


"""__________________FUNCTIONS_______________________"""


# Definir una función con parámetros de Open CV
def identify_stars_and_distance(image_path, plot=False):
    # Cargar imagen con filtro en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    imagef = image.flatten()
    # Datos de la imagen sin los pixs menores a 5 (fondo presuntamente)
    image_nob = imagef[np.where(imagef > 5)]

    # Calcular la desviación estándar (ruido de fondo) de la imagen sin el fondo
    std, median = mad_std(image), np.median(image)

    # Aplicar un umbral para binarizar la imagen y destacar los objetos (estrellas) Este parámetro se puede variar según la intensidad de la estrella
    _, thresh = cv2.threshold(image, 2 * median, 255, cv2.THRESH_BINARY)

    # Encontrar los contornos de los objetos (estrellas)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar unicamente los dos contornos más grandes detectados (asumiendo que son las estrellas)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Obtener los centros de las dos estrellas
    star_centers = []  # crear un vector para guardar los centros
    for contour in contours:
        M = cv2.moments(
            contour
        )  # calcula los momentos del contorno, función de Open CV
        if M["m00"] != 0:  # moo es el area del contorno
            cX = int(
                M["m10"] / M["m00"]
            )  # m10 es el momento espacial para calcular las coordenadas del centroide
            cY = int(
                M["m01"] / M["m00"]
            )  # m01 es el momento espacial para calcular las coordenadas del centroide
            star_centers.append(
                (cX, cY)
            )  # guardar las coordenadas de los centros encontrados

    # Calcular la distancia entre las dos estrellas
    if len(star_centers) == 2:
        pixel_distance = dist.euclidean(star_centers[0], star_centers[1])
        if plot:
            # Load the image
            img = mpimg.imread(image_path)
            # Display the image
            plt.imshow(img)
            # print(star_centers)
            plt.plot(
                [star_centers[0][0], star_centers[1][0]],
                [star_centers[0][1], star_centers[1][1]],
                "ro",
                mfc="none",
            )
            plt.tight_layout()
            plt.axis("off")
            # plt.savefig('hm.png', dpi=400)
            plt.show()

        return pixel_distance
    else:
        # print("No se encontraron estrellas.")
        return 0

    # Mostrar la imagen con los centros de las estrellas marcados
    # plt.imshow(image, cmap='gray')
    # plt.title(f"Distancia en pixeles: {pixel_distance:.2f}")
    # plt.show()


# Definir una funcion para extraer el numero del frame del nombre del archivo
def extract_frame_number(filename):
    match = re.search(r"frame(\d+)\.jpg", filename)
    return int(match.group(1)) if match else -1
"""__________________________________________________"""

"""_____________________MAIN_________________________"""
folder = r"D:\Tecnicas_observacionales\Seeing\test"  # <--------------AQUI SE CAMBIA LA RUTA DONDE SE ENCUENTRAN LOS FRAMES
files = np.sort([file for file in os.listdir(folder)])

values = np.zeros((len(files),6), dtype=object)
ff = -1

seeing = []
sigma = []
sigma_sec = []
fried = []

for filename in files:
    ff+=1

    # Abrir carpeta donde se encuentran los frames del video
    carpeta = f"{folder}/{filename}"
    print(filename)

    # Busqueda de los archivos .jpg
    archivos = glob.glob(carpeta + r"/*.jpg")

    # Ordenar los archivos numéricamente por el número de frame en el nombre asi (0, 1, 2, 3, ...)
    archivos.sort(key=extract_frame_number)

    # Definir los frames que se procesaran, se escogieron unicamente los frames entre el 50% y el 80% para eliminar errores al inicio y fin
    start = int(len(archivos)*0.78)
    end = int(len(archivos)*0.8)

    nombres = []
    for j in archivos:
        if carpeta in j:
            nombres.append(j.replace(carpeta, ""))

    if nombres != []:
        l = len(nombres)
        print(f"Se encontraron {l} frames(.jpg) se procesarán del {start} al {end}")
        print("\n :: CONSTRUYENDO EL HISTOGRAMA :: \n")
    else:
        print("Su carpeta no tiene archivos .jpg")

    distances = []
    plotit = False

    for k in range(start, end):
        # Usar la función en un frame
        dis = identify_stars_and_distance(archivos[k], plotit)

        distances.append(dis)  

        if dis!=0:
            plotit=False
    # print(distances)
    distances = np.array(distances)
    no_stars = len(np.where(distances==0)[0])
    # print(f'No se encontraron estrellas en {no_stars} frames, es decir en el {round(no_stars/(end-start+1)*100,2)}% de ellos')

    distances = distances[np.where(distances!=0)]

    # Ajuste de los datos a una distribución normal
    mu, std = norm.fit(distances)
    # Normalizar los valores del histograma
    weights = np.ones_like(distances) / len(distances)
    # Graficar el histograma con las distancias
    count, bins, ignored = plt.hist(distances, weights=weights, bins=15, alpha=0.6, color="g", edgecolor="black")

    # Graficar la función de densidad de probabilidad (PDF) ajustada
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Normalizar la PDF para que se ajuste al histograma
    bin_width = bins[1] - bins[0]
    p_normalized = p * bin_width  # Ajuste el área de la PDF al histograma
    plt.plot(x, p_normalized, "k", linewidth=2)

    """_____________________SCALE________________________"""

    # scale_size = 0.358  # segundos de arco por px ('' /px)
    scale_size = 1.12 # segundos de arco por px ('' /px)
    """__________________________________________________"""

    # Mostrar los valores de mu y sigma
    print(f"Media (mu): {mu:.3f} pixeles, Desviación estándar (σ): {std:.3f} pixeles")
    sigma_sc = std*scale_size
    print(f"Considerando el factor de escala {scale_size} la desviación estándar sigma (σ): {sigma_sc:.3f} sec")
    print(f"En la distribución normal el FWHM está dado por (2.355 * σ): {2.355*std:.3F}")

    # Parametro de Fried
    """
    Consideraciones para la ecuación del parámetro de Freid:
    206264.8 el factor de conversion de arcsec to rad ( "/rad)
    se asume un r_0 para un solo sentido, la ecuación transversal
    """
    # r_0 = (((0.358 * (lamb / std) ** 2) ** 3) / Dhole) ** (1 / 5)
    r_0 = (((sigma_sc/206264.8)**2)/((2*lamb**2)*(0.179*Dhole**(-1/3)-0.145*dsep**(-1/3))))**(-3/5)
    print(f"\nEl parámetro de Fried es :{r_0:.2f} cm \n")

    # Seeing sin corregir por cenit
    theta = 0.98 * (lamb / r_0) * 206264.8

    print(f"El valor del seeing sin la corrección al cenit es de: {theta:.3f}")

    # Grafico
    plt.title(r"$\mathrm{Histograma\ de\ distancias\ entre\ estrellas:}\ \mu=%.3f,\ \sigma=%.3f,\ \mathrm{FWHM}=%.3f$" % (mu, std, 2.355 * std))
    plt.xlabel("Distancia entre los centroides (px)")
    plt.ylabel("Frecuencia")
    plt.text(0.15, 0.9, f"Placa: {scale_size} \"/px", transform=plt.gca().transAxes, fontsize=12, color='black', ha='center')
    plt.show()

    sigma.append((filename, std))
    sigma_sec.append((filename,sigma_sc))
    fried.append((filename, r_0))
    seeing.append((filename, theta))

# Guardar los datos en un archivo .dat dentro de la misma carpeta
output_file = os.path.join(folder, "seeing_sigma.dat")

with open(output_file, "w") as f:
    f.write("# Archivo con valores de seeing, sigma, sigma_sec y fried\n")
    f.write("# Filename   Sigma(px)   Sigma(sec)   r_0(cm)   Seeing_sin_corregir\n")
    for (file_sigma, value_sigma), (file_sigma_sec, value_sigma_sec), (file_fried, value_fried), (file_seeing, value_seeing) in zip(sigma, sigma_sec, fried, seeing):
        f.write(f"{file_seeing} {value_sigma:.3f} {value_sigma_sec:.3f} {value_fried:.3f} {value_seeing:.3f}\n")

print(f"Archivo guardado en: {output_file}")
