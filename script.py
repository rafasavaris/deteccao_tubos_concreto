"""
 DETECTOR DE TUBOS DE CONCRETO EM IMAGENS UTILIZANDO 'OpenCV'
------------------------------------------------------------------
 * Autora: Rafaela F. Savaris
 * Data: 17/11/2025
 * Descrição:
    Este script detecta tubos de concreto em uma imagem. Ao final,
    exibe as detecções e imprime quantos tubos foram encontrados.
 * Requisitos:
    - Python 3.x
    - openCV (cv2)
    - numPy
    - argparse
 * Uso:
    python script.py caminho_imagem.jpg
------------------------------------------------------------------
"""

import cv2
import numpy as np
import argparse

def detectar_tubos(imagem_path):
    """
    * Detecta tubos circulares em uma imagem.

    * Parâmetros:
        imagem_path (str): Caminho da imagem a ser processada.

    * Exibe:
        - Pré-processamento
        - Máscara final
        - Resultado com tubos detectados
    """

    # Leitura da imagem
    img = cv2.imread(imagem_path)
    if img is None:
        print("Erro: imagem não encontrada.")
        return

    original = img.copy()

    # ---------- PRÉ-PROCESSAMENTO ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converte para tons de cinza

    # Kernel grande para destacar regiões escuras
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    # Operação BlackHat: realça sombras/objetos escuros
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)

    # Reforça regiões escuras juntando BlackHat e grayscale
    gray_corr = cv2.add(gray, blackhat)

    # Normalização para melhorar o contraste geral
    gray_norm = cv2.normalize(gray_corr, None, 0, 255, cv2.NORM_MINMAX)

    # CLAHE: equalização adaptativa para contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7, 7))
    gray_clahe = clahe.apply(gray_norm)

    # Suavização bilateral para reduzir ruído mantendo bordas definidas
    smooth = cv2.bilateralFilter(gray_clahe, 8, 75, 80)
    cv2.imshow("1 - Preprocessamento", smooth)

    # ---------- MÁSCARA DE REGIÕES ESCURAS ----------
    # THRESH_BINARY_INV: pixels escuros ficam brancos
    _, mask_pretos = cv2.threshold(smooth, 100, 255, cv2.THRESH_BINARY_INV)

    # Kernel menor para operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Fecha pequenos buracos dentro dos objetos
    mask_pretos = cv2.morphologyEx(mask_pretos, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove ruídos e regiões pequenas desconectadas
    mask_pretos = cv2.morphologyEx(mask_pretos, cv2.MORPH_OPEN, kernel, iterations=4)

    cv2.imshow("2 - Mascara Final", mask_pretos)

    # ---------- CONTORNOS ----------
    contornos, _ = cv2.findContours(mask_pretos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tubos = []
    resultado = original.copy()

    # ---------- FILTRAGEM ----------
    for cnt in contornos:

        # Contornos muito curtos são descartados
        if len(cnt) < 15:
            continue

        # Filtra por área aproximada do tubo
        area = cv2.contourArea(cnt)
        if area < 800 or area > 20000:
            continue

        # Perímetro necessário para calcular circularidade
        perimetro = cv2.arcLength(cnt, True)
        if perimetro == 0:
            continue

        # Circularidade: próximo de 1 indica forma circular
        circularidade = 4 * np.pi * area / (perimetro ** 2)
        if circularidade < 0.65:
            continue

        # Hull convexa: usado para medir convexidade (muito convexo não é tubo)
        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        if area_hull == 0:
            continue

        # Convexidade mínima para evitar contornos deformados
        convexidade = area / area_hull
        if convexidade < 0.82:
            continue

        # Compara perímetro original com perímetro convexo
        perimetro_hull = cv2.arcLength(hull, True)
        ratio_perimetro = perimetro / perimetro_hull
        if ratio_perimetro < 0.90:
            continue

        # Ajuste de elipse ao contorno
        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (x, y), (MA, ma), angle = ellipse

        # Evita divisão por zero
        if ma == 0 or MA == 0:
            continue

        # Verifica se a elipse não está muito alongada
        ratio_ellipse = max(MA, ma) / min(MA, ma)
        if ratio_ellipse > 2.0:
            continue

        # Se passou por todos os filtros, é considerado um tubo (adiciona à lista e desenha)
        tubos.append(ellipse)
        cv2.ellipse(resultado, ellipse, (0, 255, 0), 2)
        cv2.circle(resultado, (int(x), int(y)), 4, (0, 0, 255), -1)

    # ---------- RESULTADO FINAL ----------
    print(f"\nTubos detectados: {len(tubos)}")

    cv2.imshow("3 - Tubos Detectados", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- ENTRADA ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector de tubos pretos circulares")
    parser.add_argument("imagem", type=str, help="Caminho da imagem")
    args = parser.parse_args()

    detectar_tubos(args.imagem)