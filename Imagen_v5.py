####################################################
##########         IMAGEN_V5.PY           ##########
####################################################


# Programa que construye el histograma de frecuencias de la separación entre los
# dos "spots" en el DIMM. Utiliza el método de FloodFill para determinar los centroides.

####################################################

# ANTES DE COMPILAR, RECORDAR BORRAR LOS ARCHIVOS:
#  test.dat, out.dat out1.dat y out2.dat

####################################################

import numpy
from pylab import *

ESCALAPLACA = 1 # segundos de arco por px ('' /px)

def FloodFill(im, x, y):
    F = [] # 1. Set Q to the empty queue.
    D = []
    if (im[(x,y)] != 0):   #2. If the color of node is not equal to target-color, return. 
        F.append((x,y))   #3. Add node to Q.
        for i in F: #4. For each element n of Q:
            if (im[i] == 255):  #5.     If the color of n is equal to target-color:
                E = []
                E.append(i)
                im[i] = 0
                c = 1
                co = 1
                w = (i[0], i[1]-1)  #6.         Set w and e equal to n.
                e = (i[0], i[1]+1)
                while (im[w] == 255 ):
                    E.append(w) # Entre w y e
                    D.append(w) # Datos a retornar
                    im[w] = 0
                    c = c + 1
                    w = (x, y-c)  #7.         Move w to the west until the color of the node to the west of w no longer matches target-color.
                while (im[e] == 255):
                    E.append(e) # Entre w y e
                    D.append(e)
                    im[e] = 0    #9.         Set the color of nodes between w and e to replacement-color.
                    co = co + 1
                    e = (x, y+co) #8.         Move e to the east until the color of the node to the east of e no longer matches target-color.
                E.sort()
                for j in E:  #10.         For each node n between w and e:
                    n = (j[0]-1, j[1]) 
                    s = (j[0]+1,j[1])
                    if (im[n] == 255): #11.             If the color of the node to the north of n is target-color, add that node to Q.
                        F.append(n)
                    if (im[s] == 255): #12.             If the color of the node to the south of n is target-color, add that node to Q.
                        F.append(s)  #13. Continue looping until Q is exhausted.
                ne = (e[0]-1, e[1])  
                se = (e[0]+1, e[1])  
                nw = (w[0]-1, w[1])  
                sw = (w[0]+1, w[1])  
                if (im[ne] == 255):
                    F.append(ne)
                if (im[ne] == 255):
                    F.append(ne)
                if (im[ne] == 255):
                    F.append(ne)
                if (im[ne] == 255):
                    F.append(ne)
        D.sort()
    return im, D
     
def MassCenter(D):
    s = len(D)
    R1 = 0
    R2 = 0
    for i in D:
        R1 = R1 + i[0]
        R2 = R2 + i[1]
    R = ( R1/s , R2/s )    #Promedio aritmetico vectorial
    return R

def Area(D):
    A = len(D)       #Conteo de pixeles
    return A

def Rectangle(D):
    X = []
    Y = []
    for i in D:
        X.append(i[0])
        Y.append(i[1])
    X.sort()
    Y.sort()
    a = X[0]
    b = Y[0]
    X.reverse()
    Y.reverse()
    c = X[0]
    d = Y[0]
    P1 = (a, b)
    P2 = (a, d)
    P3 = (c, d)
    P4 = (c, b)
    return P1, P2, P3, P4

# ZONA DE LECTURA DE LOS JPGS CORRESPONDIENTES AL VIDEO #
 
import glob    
carpeta = (r'D:\Tecnicas_observacionales\Seeing\test\G-1_altair_270824_21_30_DSC_0019_mascara')
# Busqueda de los archivos .jpg
archivos = glob.glob(carpeta + r"/*.jpg")
nombres = []
for j in archivos:
  if carpeta in j:
    nombres.append(j.replace(carpeta,'')) 

if nombres != []:
  l = len(nombres)
  print(f'\n Su carpeta tiene {l} archivos .jpg \n')
  print(f'No. 1: {nombres[0]}')
  print(f'          ....            ')
  print(f'No. {l}: {nombres[l-1]}')
  print(f'          ....            ')
  print(f'          ....            ')
  print('\n :: CONSTRUYENDO EL HISTOGRAMA :: \n')
  print(f'          ....            ')

else: 
  print('\n Su carpeta no tiene archivos .jpg \n')
  
for k in range(len(archivos)):
  fits_path = archivos[k]
  image = imread(fits_path)
#  imgplot=plt.imshow(image)
  # plt.show()

  
  
  S = []
  Data = []
  objects = 0
  for i in shape(image):
      S.append( i )
      
  if ( len(S)==3 ):
      image = image[:,:,0]

  imbin = where(image<128,0,255)
 # imgplot1=plt.imshow(imbin)
  #plt.show()

  for i in range(0, S[0], 1):
      for j in range(0, S[1], 1):
          imbin, D = FloodFill(imbin, i ,j)
          if (D != []):
              Data.append(D)
              objects = objects + 1

  log_file = open("test.dat","a")  
#  print ('\n El numero de objetos en la imagen es: ', objects)       
  for i in range (0, objects, 1):
      P1, P2, P3, P4 = Rectangle( Data[i] )
 #     print ('\n Para el objeto', i+1, ':')
 #     print (' Centroide : ', MassCenter( Data[i] ))
      print (i+1, MassCenter( Data[i]), file=log_file)
 #     print (MassCenter( Data[i])[0]) #y1
 #     print (MassCenter( Data[i])[1]) #x1
#patn = re.sub(r"[\([{})\]]", "", text)
  f = open("test.dat", "r")
  fout = open("out.dat", "w")

  Lines = f.readlines()
  for line in Lines:
      dna = re.sub(r"[\([{,\n})\]]","",line)
      columns = dna.split()
      if columns[0] != "3" and  columns[0] != "4":
          spot = columns[0]
          equis = float(columns[2])
          ye = float(columns[1])
          rd_equis=round(equis,3)
          rd_ye=round(ye,3)
  
          
 #         print(spot,xuno,xdos,yuno,ydos)
    #  dt = xuno - xdos
     #     if i>=2:
                    
          
          print(spot,rd_equis,rd_ye, file=fout)
   #   print(type(dna1))
  #    print(dna)
  
  #    print(dna, file=fout)
#  fout1 = open('out1.dat', 'r')

xuno=[]
yuno=[]
xdos=[]
ydos=[]
dl = []
dt = []
Dist = []
equis = []
equis1 = []
ye = []
  
  
 
fout1 = open("out1.dat", "a")

with open("out.dat", "r") as f:
    content = f.read().splitlines()
    for i, line in enumerate(content):
        if i == len(content) - 1:
            print(line,file=fout1)
              
        else:
            print(line + content[i+1],file=fout1)      
fout1.close()

  
  
  
fread1 = open("out1.dat", "r")
fout2 = open("out2.dat", "w")

for line in fread1:
    columns = line.split()
  #  print(columns)
    if columns[0] == "1":
        xuno = float(columns[1])
        yuno = float(columns[2])
        xdos = float(columns[3])
        ydos = float(columns[4])
        dl = xuno - xdos   
        dt = yuno - ydos
        Dist = (dl**2 + dt**2)**0.5
        
        print(Dist*ESCALAPLACA, file=fout2)
    
fout2.close()

distante = []

with open("out2.dat") as efi:
    
    for line in efi:
        distante.append(float(line))

import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as pyplot
# best fit of data
mu, std = norm.fit(distante) 

#Normalise the histogram values
weights = np.ones_like(distante) / len(distante)
plt.hist(distante, weights=weights)
# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
print(mu,std)
plt.plot(x, p, 'k', linewidth=2)
#plt.text(0.35, 9, r'$\mu=mean, b=3$')
plt.title(r'$\mathrm{Histograma-MVI1346.MOV\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, std))
plt.xlabel('Distancia entre los centroides (px)')
plt.ylabel('Frequencia')


plt.show()


     
