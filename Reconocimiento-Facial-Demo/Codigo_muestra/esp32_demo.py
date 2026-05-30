ESP32_URL = "http://192.168.X.X"

def enviar_a_esp32(comando):
    if comando == '1':
        requests.get(f"{ESP32_URL}/abrir")
    elif comando == '0':
        requests.get(f"{ESP32_URL}/cerrar")