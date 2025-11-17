import cv2
import numpy as np
import argparse

def detectar_tubos_pretos(imagem_path):
    img = cv2.imread(imagem_path)
    if img is None:
        print("Erro: imagem não encontrada.")
        return
    
    original = img.copy()

    # Pré-processamento da imagem
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contraste = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = contraste.apply(gray)

    smooth = cv2.bilateralFilter(gray, 9, 150, 150)
    smooth = cv2.medianBlur(smooth, 5)

    cv2.imshow("1 - Preprocessamento", smooth)

    # Segmentação dos tubos pretos
    _, mask_pretos = cv2.threshold(smooth, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_pretos = cv2.morphologyEx(mask_pretos, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_pretos = cv2.morphologyEx(mask_pretos, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imshow("2 - Mascara Pretos", mask_pretos)

    # Detecção e filtragem de contornos
    contornos, _ = cv2.findContours(mask_pretos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tubos = []
    resultado = original.copy()

    for cnt in contornos:
        if len(cnt) < 5:
            continue

        area = cv2.contourArea(cnt)
        if area < 800 or area > 20000:
            continue

        perimetro = cv2.arcLength(cnt, True)
        if perimetro == 0:
            continue

        circularidade = 4 * np.pi * area / (perimetro**2)
        if circularidade < 0.6:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (x, y), (MA, ma), angle = ellipse
        if ma == 0:
            continue

        ratio = max(MA, ma) / min(MA, ma)
        if ratio > 7.5:
            continue

        tubos.append(ellipse)
        cv2.ellipse(resultado, ellipse, (0,255,0), 2)
        cv2.circle(resultado, (int(x), int(y)), 4, (0,0,255), -1)

    print(f"\nTubos detectados: {len(tubos)}")

    cv2.imshow("3 - Tubos Pretos Detectados", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector de tubos pretos circulares")
    parser.add_argument("imagem", type=str, help="Caminho da imagem")
    args = parser.parse_args()

    detectar_tubos_pretos(args.imagem)