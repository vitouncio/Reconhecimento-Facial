#1.1 Criar Função "Gerar pastas com faces" 
#pip install opencv-python deepface matplotlib imageio

import os
import cv2
from deepface import DeepFace

def generateFolderWithFaces(output_folder, video_path, target_image_path):
                #(Pasta onde vai ser salvo, caminho de video, Imagem de referência)

    max_saved_faces=10  # Maximo de imagens a serem salvas
    step_frame=20  # Frames a serem pulados
    detected_faces_list = []  # Lista para armazenar as faces detectadas
    detected_faces_counter = 0  # Contador de faces detectadas
    frame_counter = 0  # Contador de frames

    # Carregue a imagem alvo
    target_image = cv2.imread(target_image_path)

    # Carregue o vídeo
    cap = cv2.VideoCapture(video_path)

    # Enquanto o Video estiver aberto:
    while cap.isOpened():

        # Contabiliza frames
        frame_counter += 1

        # Pega o próximo frame
        ret, frame = cap.read()

        # Se Acabou o vídeo ou detectou o limite de faces: Pare
        if not ret or detected_faces_counter >= max_saved_faces:
            break

        # "Pula" alguns frames
        if frame_counter % step_frame != 0:
            continue

        # Detecte as faces no frame usando DeepFace.verify
        detected_faces = DeepFace.verify(frame, target_image, "VGG-Face", enforce_detection=False)

        # Se no dicionario 'verified' for verdadeiro e 'similarity_metric' for "cosine":
        if isinstance(detected_faces, dict) and detected_faces.get('verified') == True and detected_faces.get('similarity_metric') == "cosine":

            # Total de faces detectadas até o momento
            detected_faces_counter += 1

            # Pegue as coordenadas da face (se existirem)
            facial_areas = detected_faces.get('facial_areas', {})
            img1_area = facial_areas.get('img1') if isinstance(facial_areas, dict) else None

            if img1_area:
                x = int(img1_area['x'])
                y = int(img1_area['y'])
                w = int(img1_area['w'])
                h = int(img1_area['h'])

                face_crop = frame[y:y + h, x:x + w]

                # condiciona a encontrar faces apenas com esses tamanhos entre 150px a 280px
                if h >= 150 and h <= 280:
                    # Adicionar a face detectada à lista (mantemos BGR para salvar com cv2.imwrite)
                    detected_faces_list.append(face_crop)

                    # Assegura que a pasta de saída existe
                    os.makedirs(output_folder, exist_ok=True)

                    # Região das faces com o numero de faces ja encontradas e qual seu frame
                    img_path = os.path.join(output_folder, f"face_{detected_faces_counter}_{frame_counter}.jpg")

                    # Escreve o arquivo na pasta
                    cv2.imwrite(img_path, face_crop)

                    # Adicionar retângulo ao redor da face e marcações para exibir
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)

    # Fechar o vídeo após processamento
    cap.release()

    # Retornar a lista de faces detectadas
    return detected_faces_list

#1.2 Adicionar caminhos para Leon

# Caminho para o vídeo de entrada para serem retiradas as faces
video_path = os.path.join('vids', 'leon&nilce.mp4')

# Caminho para a foto do rosto da pessoa desejada
foto_path = os.path.join('imgs', 'img_Leon.jpg')

# Pasta para salvar as imagens recortadas
folder_name = os.path.join('imgs', 'Folder_leon')

# Chamada da Função
Leon_faces = generateFolderWithFaces(folder_name, video_path, foto_path)

#1.3 Adicionar caminhos para Nilce

# Caminho para o vídeo de entrada
video_path = os.path.join('vids', 'leon&nilce.mp4')

# Caminho para a foto do rosto da pessoa desejada
foto_path = os.path.join('imgs', 'img_Nilce.jpg')

# Pasta para salvar as imagens recortadas
folder_name = os.path.join('imgs', 'Folder_nilce')

Nilce_faces = generateFolderWithFaces(folder_name, video_path, foto_path)

#2.1 Função de mostrar Imagnes utiizando a matplotlib

import matplotlib.pyplot as plt

def display_images(image_folder, num_images=5):

    # Lista de nomes de arquivo das imagens na pasta
    file_names = os.listdir(image_folder)

    # Crie uma figura com subplots para exibir as imagens
    fig, axes = plt.subplots(1, num_images, figsize=(16, 16))

    # Loop para exibir as primeiras num_images imagens
    for i in range(num_images):
        # Leia a imagem do arquivo
        image_path = os.path.join(image_folder, file_names[i])
        image = cv2.imread(image_path)

        # Converta a imagem para o formato RGB (matplotlib usa RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Exiba a imagem no subplot correspondente
        axes[i].imshow(image_rgb)
        axes[i].axis('off')  # Desative os eixos

        # Defina um título para a imagem
        axes[i].set_title(f'Face {i + 1}', fontsize=12)

    # Ajuste o espaçamento entre os subplots
    plt.tight_layout()

    # Exiba as imagens
    plt.show()


# 2.1 Chamada para o Leon
folder_name = os.path.join('imgs', 'Folder_leon')
if os.path.isdir(folder_name):
    display_images(folder_name, num_images=5)

# 2.2 Chamada para a Nilce
folder_name = os.path.join('imgs', 'Folder_nilce')
if os.path.isdir(folder_name):
    display_images(folder_name, num_images=5)

    