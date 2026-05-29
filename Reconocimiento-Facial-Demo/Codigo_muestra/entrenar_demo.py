import cv2
import os
import numpy as np

# ===== CONFIGURACIÓN =====
# Se recomienda tener entre 80 y 150 fotos por alumno en sus carpetas
MAX_IMAGENES_POR_PERSONA = 80 
TAMANO_ROSTRO = (150, 150)

dataPath = 'dataset_demo'

# Inicializar el reconocedor LBPH
modelo = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)

rostros = []
ids = []
personas = []

print("="*60)
print("1. INICIANDO ENTRENAMIENTO ESTRICTO")
print("="*60)

# Detector para localizar el rostro exacto dentro de las fotos
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Obtener carpetas de alumnos
carpetas = sorted([
    d for d in os.listdir(dataPath)
    if os.path.isdir(os.path.join(dataPath, d))
])

for id_actual, carpeta in enumerate(carpetas):
    ruta_carpeta = os.path.join(dataPath, carpeta)
    print(f"\n📁 Procesando alumno: {carpeta}")
    personas.append(carpeta)
    
    imagenes = [f for f in os.listdir(ruta_carpeta) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(imagenes) > MAX_IMAGENES_POR_PERSONA:
        imagenes = imagenes[:MAX_IMAGENES_POR_PERSONA]
    
    contador_persona = 0
    
    for archivo in imagenes:
        ruta = os.path.join(ruta_carpeta, archivo)
        img = cv2.imread(ruta)
        
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_mejorado = cv2.equalizeHist(gray)
        
        # Detectar el rostro en la foto
        rostros_detectados = detector.detectMultiScale(
            gray_mejorado, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        if len(rostros_detectados) > 0:
            # Tomar el rostro más grande detectado
            (x, y, w, h) = max(rostros_detectados, key=lambda r: r[2]*r[3])
            rostro_cortado = gray[y:y+h, x:x+w]
        else:
            # Si la foto ya es un recorte perfecto y chiquito, la aceptamos
            if gray.shape[0] < 200 and gray.shape[1] < 200:
                rostro_cortado = gray
            else:
                # FILTRO: Si es una foto grande y no se ve el rostro, SE DESCARTA
                print(f"   ❌ {archivo}: Descartada (No se halló un rostro limpio)")
                continue
        
        # Normalizar tamaño y luz
        rostro_final = cv2.resize(rostro_cortado, TAMANO_ROSTRO, interpolation=cv2.INTER_CUBIC)
        rostro_final = cv2.equalizeHist(rostro_final)
        
        rostros.append(rostro_final)
        ids.append(id_actual)
        contador_persona += 1
    
    print(f"   ✅ Rostros limpios indexados: {contador_persona}")

print(f"\n📊 Total rostros listos para el modelo: {len(rostros)}")

if len(rostros) > 0:
    print("\n🔄 Generando mapa matemático del modelo...")
    modelo.train(rostros, np.array(ids))
    
    os.makedirs("modelo", exist_ok=True)
    # Guardado del modelo omitido en esta versión demo
    print("============== ¡ENTRENAMIENTO FINALIZADO CON ÉXITO! ==============")
else:
    print("❌ Error: No se encontraron rostros válidos para entrenar.")