# Detector de Tubos de Concreto em Imagens usando OpenCV

## Descrição
Este script detecta tubos de concreto em imagens.  
Ele processa a imagem, aplica filtros para realçar tubos escuros, cria uma máscara, identifica contornos e ajusta elipses para cada tubo detectado.  
Ao final, exibe as imagens com pré-processamento, máscara final e resultado com tubos detectados, e imprime no terminal o número de tubos encontrados.

---

## Requisitos
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy (`numpy`)  
- argparse  

As dependências podem ser instaladas da seguinte forma:

```bash
pip install opencv-python numpy
```

## Uso

```bash
python script.py caminho_imagem.jpg
```

onde:
* **caminho_imagem.jpg**: caminho da imagem que deseja processar.

## Observações

* O processamento é feito com a resolução original, garantindo precisão;
* Apenas a exibição é reduzida (escala=0.6) para caber melhor na tela;
* Ideal para imagens com tubos visíveis e contraste suficiente.

---