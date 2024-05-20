from flask import Flask, request, Response
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pytesseract
import os
from yolov5.models.experimental import attempt_load

app = Flask(__name__)

tesseract_dir = os.path.join(os.path.dirname(__file__), 'tesseract')
pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_dir, 'tesseract.exe')

# Cargar el modelo YOLOv5
model_path = './best.pt'

model = attempt_load(model_path)
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

@app.route('/')
def hello_world():
    return '¡Hola! Esta es una prueba de funcionamiento.'

@app.route('/detect', methods=['POST'])
def detect_objects():
    #Opciones
    daltonismo = int(request.form.get('option'))
    detectarTexto = int(request.form.get('text_detection'))
    colorBlindProtanopiaFactor = 0.0
    colorBlindDeuteranopiaFactor = 0.0
    colorBlindTritanopiaFactor = 0.0
    colorBlindDaltonizeFactor = 0.0
    accessibilityBrightnessFactor = 0.0
    accessibilityContrastFactor = 0.0
    filtrar = False

    # Obtener el fotograma del cliente
    img_encoded = request.files['video'].read()
    nparr = np.frombuffer(img_encoded, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocesar el fotograma
    image = Image.fromarray(frame)
    input_image = transform(image).unsqueeze(0)

    # Realizar la inferencia
    results = model(input_image)

    # Obtener las detecciones
    predictions = results.cpu().numpy()[0]

    # Procesar detecciones
    boxes = []
    confidences = []
    class_probs_list = []

    if predictions.size > 0:  # Verificar si hay detecciones
        for box in predictions:
            cx, cy, h, v, conf, *class_probs = box

            x1 = int(cx - (h/2))
            y1 = int(cy - (v/2))
            x2 = int(cx + (h/2))
            y2 = int(cy + (v/2))

            if conf > 0.55:
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)
                class_probs_list.append(class_probs)

        # Aplicar Non-Maximum Suppression si hay detecciones
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.5)

            # Dibujar las cajas delimitadoras restantes
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                predicted_class, max_prob = get_highest_probability_class(class_probs_list[i], model.names)
                
                if predicted_class == 'Blackboard-Whiteboards':
                    zoom_factor = max((y2 - y1) / frame.shape[0], (x2 - x1) / frame.shape[1])
                    if zoom_factor < 0.5: 
                        zoom_factor = zoom_factor + 0.4
                    
                    # Calcular el centro del bounding box original
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Calcular las nuevas dimensiones del bounding box después del zoom
                    new_width = int((x2 - x1) / zoom_factor)
                    new_height = int((y2 - y1) / zoom_factor)
                    
                    # Ajustar las coordenadas del bounding box después del zoom
                    new_x1 = int(cx - new_width / 2)
                    new_y1 = int(cy - new_height / 2)
                    new_x2 = int(cx + new_width / 2)
                    new_y2 = int(cy + new_height / 2)
                    
                    # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
                    new_x1 = max(0, new_x1)
                    new_y1 = max(0, new_y1)
                    new_x2 = min(frame.shape[1], new_x2)
                    new_y2 = min(frame.shape[0], new_y2)
                    
                    # Recortar la ROI según las nuevas coordenadas del bounding box
                    roi = frame[new_y1:new_y2, new_x1:new_x2]
                    
                    # Redimensionar la ROI
                    resized_roi = cv2.resize(roi, (frame.shape[1], frame.shape[0]))
                    frame = resized_roi

                elif predicted_class == 'Person':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{predicted_class}: {confidences[i]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if daltonismo != 0:
        filtrar = True
        if daltonismo == 1:
            colorBlindProtanopiaFactor = 1.0
            colorBlindDeuteranopiaFactor = 0.0
            colorBlindTritanopiaFactor = 0.0
        if daltonismo == 2:
            colorBlindProtanopiaFactor = 0.0
            colorBlindDeuteranopiaFactor = 1.0
            colorBlindTritanopiaFactor = 0.0
        if daltonismo == 3:
            colorBlindProtanopiaFactor = 0.0
            colorBlindDeuteranopiaFactor = 0.0
            colorBlindTritanopiaFactor = 1.0

        colorBlindDaltonizeFactor = 0.9

    if filtrar:
        frame = accessibility_post_processing(frame, colorBlindProtanopiaFactor, colorBlindDeuteranopiaFactor, colorBlindTritanopiaFactor, 
                                                    colorBlindDaltonizeFactor, accessibilityBrightnessFactor, accessibilityContrastFactor)
    if detectarTexto:
        frame = detectarTextoOCR(frame)

    ret, jpeg = cv2.imencode('.jpg', frame)

    return Response(response=jpeg.tobytes(), content_type='image/jpeg')

def get_highest_probability_class(class_probs, class_names):
  max_prob = max(class_probs)
  max_prob_index = class_probs.index(max_prob)
  return class_names[max_prob_index], max_prob

# ----------------------------------------------------
# Filtros de daltonismo
# ----------------------------------------------------

def accessibility_post_processing(image, color_blind_protanopia_factor, color_blind_deuteranopia_factor, color_blind_tritanopia_factor,
                                    color_blind_daltonize_factor, accessibility_brightness_factor, accessibility_contrast_factor):
    # Normalizar la imagen
    image_normalized = image / 255.0

    color_lms = rgb_to_lms(image_normalized)

    color_lms_corrected = daltonize(color_lms, color_blind_protanopia_factor, color_blind_deuteranopia_factor, color_blind_tritanopia_factor)

    processed_image = lms_to_rgb(color_lms_corrected)
    processed_image = ((processed_image - 0.5) * (1.0 + accessibility_contrast_factor)) + 0.5
    processed_image += accessibility_brightness_factor + 0.08 * color_blind_daltonize_factor
    processed_image = np.clip(processed_image, 0.0, 1.0)

    processed_image_rgb = (processed_image * 255).astype(np.uint8)

    return processed_image_rgb


def rgb_to_lms(rgb):
    lms_matrix = np.array([[17.8824, 43.5161, 4.11935],
                           [3.45565, 27.1554, 3.86714],
                           [0.0299566, 0.184309, 1.46709]])

    lms = np.dot(rgb, lms_matrix.T)
    return lms

def lms_to_rgb(lms):
    lms_to_rgb_matrix = np.array([[0.080944, -0.130504, 0.116721],
                                  [-0.0102485, 0.0540193, -0.113615],
                                  [-0.000365294, -0.00412163, 0.693513]])

    rgb = np.dot(lms, lms_to_rgb_matrix.T)
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb

def daltonize(color, color_blind_protanopia_factor, color_blind_deuteranopia_factor, color_blind_tritanopia_factor):
    protanopia_correction = np.array([[0, 2.02344, -2.52581],
                                      [0, 1, 0],
                                      [0, 0, 1]])
    color_protanopia = np.dot(color, protanopia_correction.T) * color_blind_protanopia_factor

    deuteranopia_correction = np.array([[1, 0, 0],
                                         [0.494207, 0, 1.24827],
                                         [0, 0, 1]])
    color_deuteranopia = np.dot(color, deuteranopia_correction.T) * color_blind_deuteranopia_factor

    tritanopia_correction = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [-0.395913, 0.801109, 0]])
    color_tritanopia = np.dot(color, tritanopia_correction.T) * color_blind_tritanopia_factor
    color_corrected = (color_protanopia + color_deuteranopia + color_tritanopia) / (color_blind_protanopia_factor + color_blind_deuteranopia_factor + color_blind_tritanopia_factor)
    
    return color_corrected

# ----------------------------------------------------
# Detector de Texto (OCR)
# ----------------------------------------------------

def detectarTextoOCR(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    texto_detectado = pytesseract.image_to_string(thresh)
    
    cv2.putText(frame, texto_detectado, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame

# ----------------------------------------------------
# Inicio del servidor
# ----------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
