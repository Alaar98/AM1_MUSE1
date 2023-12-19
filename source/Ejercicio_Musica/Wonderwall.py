
nombre_archivo = "C:\\Users\\alejandro\\Documents\\GitHub\\AM1_orbits\\source\\Ejercicio_Musica\\Wonderwall_lyrics_notes.txt"


with open(nombre_archivo, "r") as archivo:
        contenido = archivo.read()
        # print("Contenido del archivo:")
        # print(contenido)


cancion = list(contenido.split("\n"))
print(cancion[0])
lyrics = list ()
acordes = list()


for i in range(len(cancion)):
     if i % 2 == 0:
         acordes.append(cancion[i])
     else:
         lyrics.append(cancion[i])


# Crear una lista de palabras distintas
palabras_distintas = list(set(lyrics))

# Crear un diccionario de recuento de palabras repetidas
# diccionario_de_recuento = {}
# for palabra in palabras_distintas:
#     conteo = lyrics.count(palabra)
#     if conteo > 1:
#         diccionario_de_recuento[palabra] = conteo

# Imprimir la lista de palabras distintas
print("Palabras distintas:", palabras_distintas)

# Imprimir el diccionario de recuento de palabras repetidas
#print("Diccionario de recuento:", diccionario_de_recuento)
