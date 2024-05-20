# import cv2
# import requests
# import numpy as np

# def enviar_video_en_vivo():
#     cap = cv2.VideoCapture(0)  # Abrir la cámara en vivo (cambiar el número si hay más de una cámara)
#     url = 'http://127.0.0.1:5000/detect'  # Dirección local del servidor Flask

#     while True:
#         ret, frame = cap.read()  # Leer un fotograma de la cámara en vivo
#         _, img_encoded = cv2.imencode('.jpg', frame)  # Codificar el fotograma como JPEG

#         # Enviar el fotograma codificado a la API
#         response = requests.post(url, 
#                                         files={'video': ('video.jpg', img_encoded.tostring(), 'image/jpeg')},
#                                          data={'option': 0, 'text_detection': 0})
        
#         # Mostrar el video procesado en una ventana de OpenCV
#         nparr = np.frombuffer(response.content, np.uint8)
#         processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # Mostrar el video procesado en una ventana de OpenCV
#         cv2.namedWindow('Video en Tiempo Real', cv2.WINDOW_NORMAL)  # Crear una ventana con capacidad de cambio de tamaño
#         cv2.setWindowProperty('Video en Tiempo Real', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Establecer la ventana en pantalla completa
#         cv2.imshow('Video en Tiempo Real', processed_frame)

#         # Salir del bucle si se presiona la tecla 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     enviar_video_en_vivo()
import tkinter as tk
from tkinter import ttk, IntVar
import cv2
import requests
import numpy as np
import threading
from PIL import Image, ImageTk

class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación Asistencia Visual")

        # Establecer la ventana en pantalla completa
        self.root.attributes('-fullscreen', True)

        # Barra de menú
        self.menu_bar = tk.Menu(root)
        self.root.config(menu=self.menu_bar)

        # Menú 'Opciones'
        self.opciones_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Opciones", menu=self.opciones_menu)
        
        # Submenús de 'Opciones'
        self.opciones_menu.add_command(label="Accesibilidad", command=self.mostrar_pestaña_daltonismo)
        self.opciones_menu.add_command(label="Cerrar Aplicación", command=self.cerrar_aplicacion)

        # Label para mostrar la imagen
        self.label_imagen = ttk.Label(root)
        self.label_imagen.pack(expand=True, fill="both")

        # Pestaña de Daltonismo (inicialmente oculta)
        self.tab_daltonismo = tk.Canvas(root, bg="gray", bd=0, highlightthickness=0)
        self.tab_daltonismo.place(x=50, y=50, width=200, height=200)
        self.tab_daltonismo.bind("<Button-1>", self.ocultar_pestaña_daltonismo)
        self.tab_daltonismo.place_forget()

        self.tab_daltonismo_texto = self.tab_daltonismo.create_text(60, 20, text="Daltonismo", fill="white", font=("Arial", 16), anchor="n")
        self.option_var = tk.StringVar()
        self.option_var.set(0)
        tk.Radiobutton(self.tab_daltonismo, text="Ninguno", variable=self.option_var, value=0).place(x=50, y=50)
        tk.Radiobutton(self.tab_daltonismo, text="Protanopia", variable=self.option_var, value=1).place(x=50, y=80)
        tk.Radiobutton(self.tab_daltonismo, text="Deuteranopia", variable=self.option_var, value=2).place(x=50, y=110)
        tk.Radiobutton(self.tab_daltonismo, text="Tritanopia", variable=self.option_var, value=3).place(x=50, y=140)

        # Opción de detección de texto
        self.tab_deteccion_texto = self.tab_daltonismo.create_text(100, 170, text="Detección de Texto", fill="white", font=("Arial", 16), anchor="n")
        self.text_detection_var = IntVar()
        self.text_detection_var.set(0)
        self.text_detection_cb = tk.Checkbutton(self.tab_daltonismo, text="Detección de Texto", variable=self.text_detection_var)
        self.text_detection_cb.place(x=50, y=200)

        # Iniciar hilo para el video en vivo
        self.thread = threading.Thread(target=self.enviar_video_en_vivo)
        self.thread.daemon = True
        self.thread.start()

        # Bandera para controlar si la ventana está en pantalla completa
        self.fullscreen = True
        self.popup_exist = False

    # Funciones para mostrar y ocultar las pestañas
    def mostrar_pestaña_daltonismo(self):
        self.tab_daltonismo.place(x=50, y=50)

    def ocultar_pestaña_daltonismo(self, event):
        self.tab_daltonismo.place_forget()

    # Función para enviar el video en vivo y procesado
    def enviar_video_en_vivo(self):
        cap = cv2.VideoCapture(0)
        url = 'https://tfm-719t.onrender.com/detect'

        cam_connected = False

        while not cam_connected:
            ret, frame = cap.read() 

            if not ret:
                if not self.popup_exist:
                    self.mostrar_mensaje_emergente("Esperando conexión con cámara...")
                continue

            ret, frame = cap.read() 
            _, img_encoded = cv2.imencode('.jpg', frame)

            option_value = int(self.option_var.get())
            text_detection_value = int(self.text_detection_var.get())
            try:
                print("Enviando solicitud al servidor...")
                response = requests.post(url, files={'video': ('video.jpg', img_encoded.tostring(), 'image/jpeg')},
                                        data={'option': option_value, 'text_detection': text_detection_value})
                print("Respuesta del servidor:", response)
                response.raise_for_status()  # Esto generará una excepción si la solicitud falla
            except requests.exceptions.RequestException as e:
                print("Error al enviar la solicitud:", e)
                if not self.popup_exist:
                    self.popup_exist = True
                    self.mostrar_mensaje_emergente("Error al enviar la solicitud al servidor.")
                continue

            # Verificar si la respuesta es válida
            if response.status_code == 200:
                print("Respuesta recibida del servidor.")
                # Continuar procesando la respuesta...
            else:
                print("Error: Respuesta inesperada del servidor:", response.status_code)
                if not self.popup_exist:
                    self.popup_exist = True
                    self.mostrar_mensaje_emergente("Respuesta inesperada del servidor: {}".format(response.status_code))
                continue

            # Mostrar el video procesado en la interfaz
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Redimensionar la imagen al tamaño de la pantalla
            img = Image.fromarray(img)
            img = img.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
            img_tk = ImageTk.PhotoImage(image=img)

            # Actualizar la imagen en el Label
            self.label_imagen.configure(image=img_tk)
            self.label_imagen.image = img_tk

            # Cambiar a pantalla completa si no está en pantalla completa
            if not self.fullscreen:
                self.root.attributes('-fullscreen', True)
                self.fullscreen = True

            if not self.root.winfo_exists():
                break

        cap.release()

    def mostrar_mensaje_emergente(self, mensaje):
        self.popup = tk.Toplevel()
        self.popup.title("Error")
        self.popup.geometry("400x200")
        label = tk.Label(self.popup, text=mensaje, font=("Arial", 18))
        label.pack(side="top", fill="x", pady=20)
        button = tk.Button(self.popup, text="Aceptar", command=self.destroy_popup)
        button.pack()
    
    def destroy_popup(self):
        self.popup.destroy()
        self.popup_exist = False

    def cerrar_aplicacion(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.mainloop()