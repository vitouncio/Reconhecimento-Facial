# Reconhecimento Facial com DeepFace

#1.1 Carregar o video

import cv2
import os
import imageio
from deepface import DeepFace

# caminhos usando os.path.join para evitar escape sequences
video_filename = os.path.join("vids", "leon&nilce.mp4")
os.makedirs(os.path.join("vids"), exist_ok=True)

video_capture_input = cv2.VideoCapture(video_filename)

# Defina o arquivo de vídeo de saída (fallback de fps)
fps = int(video_capture_input.get(cv2.CAP_PROP_FPS)) if video_capture_input.get(cv2.CAP_PROP_FPS) else 24
output_path = os.path.join("vids", "output.mp4")
video_capture_output = imageio.get_writer(output_path, fps=fps)

# Imagens de referência (use os.path.join)
target_image_leon = os.path.join("imgs", "img_Leon.jpg")
target_image_nilce = os.path.join("imgs", "img_Nilce.jpg")

frame_counter = 0

while True:
    success, frame_bgr = video_capture_input.read()
    frame_counter += 1
    print("Frame: " + str(frame_counter))

    if not success:
        break

    # copie para RGB apenas para DeepFace; mantenha BGR para desenhar com OpenCV
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Use detector_backend='retinaface' para evitar dependência do Haarcascade do OpenCV
    try:
        detected_face_leon = DeepFace.verify(frame_rgb, target_image_leon, model_name="VGG-Face", enforce_detection=False, detector_backend="retinaface")
    except Exception as e:
        detected_face_leon = {}
    face_area_leon = detected_face_leon.get('facial_areas', {}).get('img1')

    try:
        detected_face_nilce = DeepFace.verify(frame_rgb, target_image_nilce, model_name="VGG-Face", enforce_detection=False, detector_backend="retinaface")
    except Exception as e:
        detected_face_nilce = {}
    face_area_nilce = detected_face_nilce.get('facial_areas', {}).get('img1')

    # se existir área facial, extraia coordenadas com segurança
    if detected_face_leon.get('verified') and detected_face_leon.get('similarity_metric') == "cosine" and face_area_leon:
        x_leon = int(face_area_leon['x']); y_leon = int(face_area_leon['y'])
        w_leon = int(face_area_leon['w']); h_leon = int(face_area_leon['h'])
        recognized_name = "Leon"
        print("Leon reconhecido")
        cv2.rectangle(frame_bgr, (x_leon, y_leon), (x_leon + w_leon, y_leon + h_leon), (0, 255, 0), 4)
        cv2.putText(frame_bgr, recognized_name, (x_leon, y_leon - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if detected_face_nilce.get('verified') and detected_face_nilce.get('similarity_metric') == "cosine" and face_area_nilce:
        x_nilce = int(face_area_nilce['x']); y_nilce = int(face_area_nilce['y'])
        w_nilce = int(face_area_nilce['w']); h_nilce = int(face_area_nilce['h'])
        if 150 <= h_nilce <= 250:
            recognized_name = "Nilce"
            print("Nilce reconhecida")
            cv2.rectangle(frame_bgr, (x_nilce, y_nilce), (x_nilce + w_nilce, y_nilce + h_nilce), (0, 255, 0), 4)
            cv2.putText(frame_bgr, recognized_name, (x_nilce, y_nilce - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Antes de salvar com imageio, converta para RGB
    try:
        video_capture_output.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print("Erro ao escrever frame:", e)

#2.2 Libere os objetos de captura e gravacao de video

video_capture_input.release()
video_capture_output.close()