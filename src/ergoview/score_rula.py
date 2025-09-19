import numpy as np

def calculate_rula(
    angle: float,
    elbow_y: float,
    shoulder_y: float,
    nose_y: float,
    wrist_y: float,
) -> int:
    """
    Calcula a pontuação RULA com base no ângulo do cotovelo e posições relativas do braço e ombro.

    A pontuação considera:
    - Ângulo do braço (ombro-cotovelo-pulso)
    - Correção para braço abaixado (posição de "sentido")
    - Ajustes para ombro elevado e ombro abduzido

    Parâmetros
    ----------
    angle : float
        Ângulo entre ombro, cotovelo e pulso.
    elbow_y : float
        Coordenada Y do cotovelo.
    shoulder_y : float
        Coordenada Y do ombro.
    nose_y : float
        Coordenada Y do nariz.

    Retorna
    -------
    int
        Pontuação RULA calculada.

    Exemplos
    --------
    >>> calculate_rula(160, 1.0, 1.5, 1.3)
    1
    >>> calculate_rula(160, 2.0, 1.5, 1.3)
    4
    >>> calculate_rula(90, 2.0, 1.5, 1.3)
    2
    >>> calculate_rula(50, 2.0, 1.5, 1.3)
    1
    """
    # Determina pontuação base pelo ângulo
    if angle >= 150:
        score = 4
    elif angle >= 100:
        score = 3
    elif angle >= 60:
        score = 2
    else:
        score = 1

    # Correção: se o braço estiver relaxado para baixo, reduz a pontuação
    if angle >= 150 and elbow_y > shoulder_y:
        return 1

    # Ajustes adicionais
    extra_points = 0

    raised_shoulder = 0
    abducted_shoulder = 0

    # Ombro elevado: se o ombro estiver muito próximo do nariz
    if (shoulder_y - nose_y) < 25 and shoulder_y < 270:
        raised_shoulder = 1

        extra_points += 1

    # Ombro abduzido: se o cotovelo estiver na mesma altura ou acima do ombro
    if elbow_y <= shoulder_y:
        abducted_shoulder = 1
        extra_points += 1

    if wrist_y > 450 and elbow_y >= shoulder_y:
        extra_points += 1

    return score + extra_points

def pulse(
    wrist_x: float,
    wrist_y: float,
    finger_mcp_x: float,
    finger_mcp_y: float,
    pinky_mcp_x: float,
    pinky_mcp_y: float,
) -> int:
    """
    Avalia a postura do punho com base em coordenadas em pixels da imagem.
    Calcula flexão/extensão, desvio radial/ulnar e rotação (supinação/pronação),
    e imprime os resultados com os ângulos e pontuação final.
    """

    score = 0

    # Vetores (em pixels)
    vetor_mcp = np.array([pinky_mcp_x, pinky_mcp_y]) - np.array([finger_mcp_x, finger_mcp_y])
    vetor_palma = np.array([wrist_x, wrist_y]) - np.array([finger_mcp_x, finger_mcp_y])

    # ----- 1. Flexão / Extensão -----
    eixo_vertical = np.array([0, -1])  # eixo "para cima" (menores valores de Y)
    angulo_rad = np.arccos(
        np.dot(vetor_palma, eixo_vertical) / (np.linalg.norm(vetor_palma))
    )
    angulo_flex = abs(np.degrees(angulo_rad) - 160)

    #print(f"[Flexão] Ângulo: {angulo_flex:.2f}°")

    if angulo_flex <= 2:
        #print("→ Postura neutra (0°) → +1 ponto")
        score += 1
    elif 2 < angulo_flex <= 15:
        #print("→ Flexão leve (≤15°) → +2 pontos")
        score += 2
    else:
        #print("→ Flexão acentuada (>15°) → +3 pontos")
        score += 3

    # ----- 2. Desvio radial/ulnar (horizontal) -----
    #print(f"[Desvio] wrist_x: {wrist_x:.1f}, finger_mcp_x: {finger_mcp_x:.1f}, pinky_mcp_x: {pinky_mcp_x:.1f}")
    if wrist_x < finger_mcp_x:
        #print("→ Desvio ulnar → +1 ponto")
        score += 1
    elif wrist_x > pinky_mcp_x:
        #print("→ Desvio radial → +1 ponto")
        score += 1
    #else:
        #print("→ Sem desvio → +0 ponto")

    # ----- 3. Giro (supinação / pronação) -----
    eixo_horizontal = np.array([1, 0])
    angulo_rotacao_rad = np.arccos(
        np.dot(vetor_mcp, eixo_horizontal) / (np.linalg.norm(vetor_mcp))
    )
    angulo_rotacao = np.degrees(angulo_rotacao_rad)

    #print(f"[Rotação] Ângulo palma (5–17): {angulo_rotacao:.2f}°")

    if 10 < angulo_rotacao < 170:
        #print("→ Supinação ou pronação → +2 pontos")
        score += 2
    #elif angulo_rotacao < 10:
        #print("→ Giro neutro (palma lateral) → +1 ponto")
    #else:
        #print("Ponto neutro")

    #print(f"★ Pontuação total do punho: {score}\n")
    return score
