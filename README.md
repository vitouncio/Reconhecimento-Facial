# Reconhecimento Facial — Minicurso SECOMP 2023

Repositório com exemplos de reconhecimento facial usando DeepFace e OpenCV.

## Pré-requisitos
- Python 3.8 — 3.12 (ou via Anaconda)
- Git (opcional)
- Pacotes Python listados em [requirements.txt](requirements.txt)

## Instalação (recomendada)
1. Clone o repositório (opcional):
   git clone <seu-repositorio>
   cd <seu-repositorio>

2. Crie e ative um ambiente virtual:
   - Linux / macOS:
     python3 -m venv .venv
     source .venv/bin/activate
   - Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
   - Windows (cmd):
     .\.venv\Scripts\activate.bat

3. Atualize pip e instale dependências:
   pip install --upgrade pip
   pip install -r requirements.txt

Observação: se quiser suporte GPU para TensorFlow, instale a versão compatível do TensorFlow conforme a documentação oficial.

## Instalação (opcional com Anaconda / Miniconda)
Se preferir usar Anaconda/Miniconda para gerenciar ambientes e dependências, siga uma das opções abaixo.

1. Crie e ative um ambiente (exemplo com Python 3.10):
   - PowerShell / cmd (Windows) ou terminal (Linux/macOS):
     conda create -n reconf python=3.10 -y
     conda activate reconf

2. Instale dependências básicas via conda/pip:
   - Recomendo instalar pacotes pesados por conda-forge e o restante por pip:
     conda install -c conda-forge opencv imageio matplotlib -y
     pip install --upgrade pip
     pip install -r requirements.txt

   - Alternativa (tentar instalar tudo pelo conda quando possível):
     conda install -c conda-forge opencv imageio matplotlib tensorflow -y
     pip install deepface

3. Observações sobre TensorFlow / GPU:
   - Para suporte GPU, siga a documentação oficial do TensorFlow para instalar a versão adequada (cuDNN, CUDA, drivers). Muitas vezes é melhor instalar TensorFlow com conda quando disponível:
     conda install -c conda-forge tensorflow

4. Verifique instalação:
   python -c "import deepface, cv2, imageio, matplotlib; print('OK')"

## Estrutura principal
- scripts:
  - [src/Facial_Recognition.py](src/Facial_Recognition.py) — exemplo de processamento de vídeo frame-a-frame.
  - [src/genarate_folder_with_images.py](src/genarate_folder_with_images.py) — função `generateFolderWithFaces` e `display_images` para extrair faces e mostrar imagens.
  - [src/frame_counter.py](src/frame_counter.py) — script simples para verificar FPS e contagem de frames.
- pastas de mídia:
  - `vids/` — coloque os vídeos (ex.: `leon&nilce.mp4`)
  - `imgs/` — imagens de referência (ex.: `img_Leon.jpg`, `img_Nilce.jpg`) e pastas geradas

## Como executar
- Verificar frames / FPS:
  python src/frame_counter.py

- Gerar pastas com faces (usa `generateFolderWithFaces`):
  python src/genarate_folder_with_images.py

- Processar vídeo com DeepFace e gerar saída:
  python src/Facial_Recognition.py

Cada script grava/usa arquivos nas pastas `vids/` e `imgs/`. Ajuste os nomes de arquivo dentro dos scripts conforme necessário.

## Dicas e resolução de problemas
- Se o DeepFace reclamar de detector, experimente alterar `detector_backend` (ex.: `"retinaface"`).
- Problemas de dependências (compilação de pacotes nativos) podem exigir pacotes do sistema (ex.: build-essential, libglib2.0-0, etc.) dependendo do SO.
- Se o processamento estiver muito lento, considere reduzir resolução do vídeo antes de processar ou usar GPU.

## Licença
Conteúdo do repositório — sem licença especificada. Adicione uma licença se desejar compartilhamento aberto.