"""
1. VIDEO TO FRAMES

Esta parte del código requiere la ruta donde se aloja el video, y genera los frames de los videos para el Método 1 (desenfocando)
y Método 2(prisma) del DIMM, no es necesario recortar los videos, o eliminar los frames (el código hace una limpieza  y no procesa 
los primeros y los últimos frames de cada video, es decir tomar solo los frames entre el 40% y el 80 % del video para evitar movimientos 
al principio o al final)

"""
# Solo cambiar la ruta donde se encuentran todos los videos a procesar (linea 15), el código genera una carpeta para cada video procesado

import cv2
import os
# Carpeta con los videos a los que se les quiere extraer los frames
folder_path = r"D:\Tecnicas_observacionales\Seeing\Test_videos"
filenames = [file for file in os.listdir(folder_path)]
for filename in filenames:
    filename = filename[:-4]
    print(filename)
    video_path = os.path.join(folder_path, f"{filename}.MOV")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0] # Extraer el nombre del archivo para usar como nombre de la carpeta
    
    output_folder = os.path.join(folder_path, "frames", video_name) # Crear una carpeta con el nombre del video si no existe
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
        except OSError:
            print("Error: Creating directory of data")
            exit(1)

    # Abrir el video
    cam = cv2.VideoCapture(video_path)

    # Inicializar el contador de frames
    currentframe = 0

    print(f"\n Inicio de la extracción de frames")

    while True:
        # Leer un frame del video
        ret, frame = cam.read()

        if ret:
            # Si hay frames disponibles, continuar creando imágenes
            name = os.path.join(output_folder, f"frame{currentframe}.jpg")
            #print(f"Creating... {name}") #Mostrar los frames creados

            # Guardar la imagen extraída
            cv2.imwrite(name, frame)

            # Incrementar el contador de frames
            currentframe += 1
        else:
            print("\n Extracción de frames completada")
            print(f"\n {currentframe} frames generados")
            print(f"          ....            ")
            break

    # Liberar todos los recursos y cerrar ventanas una vez terminado
    cam.release()
    cv2.destroyAllWindows()
