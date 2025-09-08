# Work Flow for Kaggle Competition
- While the Kaggle competition, I have to go through the project without Internet option
- Actually, implementaion or refactoring was not so much but setting for new environment was much harder.
- This is the short TODO list "How can I set my Kaggle notebooks working for without-internet mode."

## Asset Collecting Process
- make new notebook to collecting dependencies whl
  ### cell [1]
  ```
  !python --version
  import torch
  print(f"PyTorch Version: {torch.version}")
  print(f"CUDA Version: {torch.version.cuda}")
  ```
  ### cell [2]
  ```
  import os
  WHEELS_DIR = './offline_wheels'
  os.makedirs(WHEELS_DIR, exist_ok=True)
  !pip download torch-molecule -d {WHEELS_DIR}
  !pip download transformers -d {WHEELS_DIR}
  ```
  ### cell [3]
  ```
  PYTORCH_VERSION = "2.1.0"
  CUDA_VERSION = "cu118"
  !pip download torch-scatter -f https://data.pyg.org/whl/torch-{PYTORCH_VERSION}+{CUDA_VERSION}.html -d {WHEELS_DIR}
  ```
- run and save the notebook then you can download it or create a new dataset
- make new dataset from your output

## Dependencies Check
- Add your dataset and then do bellow
  ```
  import os
  for dirname, _, filenames in os.walk('/kaggle/input'):
      for filename in filenames:
          print(os.path.join(dirname, filename))

  # 새로 만든 데이터셋 경로로 수정해야 합니다.
  dependency_path = '/kaggle/input/neurips2025-dependencies/offline_wheels'

  # 다시 설치를 시도합니다.
  !pip install --no-index --find-links={dependency_path} torch-molecule
  ```
- if you find depedency error, then you can download the datasets and edditing
