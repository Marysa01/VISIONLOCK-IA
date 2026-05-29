from flask import Flask, render_template
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

#Detector

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

#predict

id_, confianza = modelo.predict(rostro)

if confianza < 95:
    print("Acceso permitido")
else:
    print("Acceso denegado")
    
#ESP32
ESP32_IP = "192.168.X.X"