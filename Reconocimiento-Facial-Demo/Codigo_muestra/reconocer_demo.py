id_predicho, distancia = face_recognizer.predict(rostro)

if distancia < UMBRAL_EXISTENCIA:
    nombre = imagePaths[id_predicho]
    print(f"Acceso permitido: {nombre}")
else:
    print("Usuario desconocido")