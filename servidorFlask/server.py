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
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
model.eval()

protanopia_correction = np.array([[0, 2.02344, -2.52581],
                                [0, 1, 0],
                                [0, 0, 1]])

deuteranopia_correction = np.array([[1, 0, 0],
                                    [0.494207, 0, 1.24827],
                                    [0, 0, 1]])

tritanopia_correction = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [-0.395913, 0.801109, 0]])

@app.route('/', methods=['GET'])
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
    predictions = results[0].cpu().numpy()[0]

    # Procesar detecciones
    boxes = []
    confidences = []
    class_probs_list = []

    if predictions.size > 0:  # Verificar si hay detecciones
        for box in predictions:
            cx, cy, h, v, conf, *class_probs = box
            if conf >= 0.5:
                cy = cy - 100
                x1 = int(cx - (h/2))
                y1 = int(cy - (v/2))
                x2 = int(cx + (h/2))
                y2 = int(cy + (v/2))
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)
                class_probs_list.append(class_probs)

        # Aplicar Non-Maximum Suppression si hay detecciones
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.45)

            # Dibujar las cajas delimitadoras restantes
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                predicted_class, max_prob = get_highest_probability_class(class_probs_list[i], model.names)
            
                if predicted_class == 'Blackboard-Whiteboards':
                    # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Recortar la ROI según las nuevas coordenadas del bounding box
                    roi = frame[y1:y2, x1:x2]
                    
                    # Redimensionar la ROI
                    resized_roi = cv2.resize(roi, (frame.shape[1], frame.shape[0]))
                    frame = resized_roi
                    frame = improve_image_quality(frame)

                elif predicted_class == 'Person':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{predicted_class}: {confidences[i]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if daltonismo != 0:
        filtrar = True
        if daltonismo == 1:
            colorBlindProtanopiaFactor = 0.2
            colorBlindDeuteranopiaFactor = 0.0
            colorBlindTritanopiaFactor = 0.0
        if daltonismo == 2:
            colorBlindProtanopiaFactor = 0.0
            colorBlindDeuteranopiaFactor = 0.2
            colorBlindTritanopiaFactor = 0.0
        if daltonismo == 3:
            colorBlindProtanopiaFactor = 0.0
            colorBlindDeuteranopiaFactor = 0.0
            colorBlindTritanopiaFactor = 0.5

        colorBlindDaltonizeFactor = 0.3

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

def improve_image_quality(frame):
    smoothed_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_frame = cv2.filter2D(smoothed_frame, -1, kernel)

    alpha = 1.2
    beta = 20 
    adjusted_frame = cv2.convertScaleAbs(sharpened_frame, alpha=alpha, beta=beta)

    return adjusted_frame

# ----------------------------------------------------
# Filtros de daltonismo
# ----------------------------------------------------

def accessibility_post_processing(image, color_blind_protanopia_factor, color_blind_deuteranopia_factor, color_blind_tritanopia_factor,
                                    color_blind_daltonize_factor, accessibility_brightness_factor, accessibility_contrast_factor):

    image_normalized = image / 255.0

    color_lms = rgb_to_lms(image_normalized)
    color_lms_corrected = daltonize(color_lms, color_blind_protanopia_factor, color_blind_deuteranopia_factor, color_blind_tritanopia_factor)
    processed_image = lms_to_rgb(color_lms_corrected)

    processed_image = adjust_contrast_and_brightness(processed_image, accessibility_contrast_factor, accessibility_brightness_factor, color_blind_daltonize_factor)

    processed_image = np.clip(processed_image, 0.0, 1.0)
    processed_image_rgb = (processed_image * 255).astype(np.uint8)

    return processed_image_rgb

def adjust_contrast_and_brightness(image, contrast_factor, brightness_factor, daltonize_factor):

    image = ((image - 0.5) * (1.0 + contrast_factor)) + 0.5

    image = np.clip(image, 0.0, 1.0)

    if brightness_factor != 0.0:
        image += brightness_factor

    if daltonize_factor != 0.0:
        image += daltonize_factor * 0.112

    image = np.clip(image, 0.0, 1.0)

    return image

def rgb_to_lms(rgb):
    lms_matrix = np.array([[17.8824, 43.5161, 4.11935],
                           [3.45565, 27.1554, 3.86714],
                           [0.0299566, 0.184309, 1.46709]])

    lms = np.dot(rgb, lms_matrix)
    return lms

def lms_to_rgb(lms):
    lms_to_rgb_matrix = np.array([[ 0.0809, -0.1305,  0.1167],
                                  [-0.0102,  0.0540, -0.1136],
                                  [ 0.0004, -0.0041,  0.6935]])

    rgb = np.dot(lms, lms_to_rgb_matrix)

    return rgb

def daltonize(color, color_blind_protanopia_factor, color_blind_deuteranopia_factor, color_blind_tritanopia_factor):

    if color_blind_protanopia_factor > 0:
        color_protanopia = np.dot(color, protanopia_correction) * color_blind_protanopia_factor
        return color_protanopia

    if color_blind_deuteranopia_factor > 0:
        color_deuteranopia = np.dot(color, deuteranopia_correction) * color_blind_deuteranopia_factor
        return color_deuteranopia

    if color_blind_tritanopia_factor > 0:
        color_tritanopia = np.dot(color, tritanopia_correction) * color_blind_tritanopia_factor
        return color_tritanopia

    return color

# ----------------------------------------------------
# Detector de Texto (OCR)
# ----------------------------------------------------

def detectarTextoOCR(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.medianBlur(thresh, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    texto_detectado = pytesseract.image_to_string(thresh, config=config)
    
    texto_detectado = texto_detectado.strip()

    cv2.putText(frame, texto_detectado, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame

# ----------------------------------------------------
# Inicio del servidor
# ----------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
