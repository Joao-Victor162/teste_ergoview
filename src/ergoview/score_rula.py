import numpy as np

def calculate_rula(
    angle: float,
    elbow_y: float,
    shoulder_y: float,
) -> int:
    """
    Calcula a pontuação RULA com base no ângulo do cotovelo e posições relativas do braço e ombro.
    """

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

    return score


# ==============================
# ESTADO GLOBAL DE REPETIÇÕES
# ==============================
movement_states = {
    "abducted": {
        "rep_counter": 0,
        "bonus_given": False,
        "in_motion": False,
        "first_point_given": False,
        "cooldown_counter": 0,
    },
    "elevated": {
        "rep_counter": 0,
        "bonus_given": False,
        "in_motion": False,
        "first_point_given": False,
        "cooldown_counter": 0,
        "history": [],
    },
    "supported": {
        "rep_counter": 0,
        "bonus_given": False,
        "in_motion": False,
        "first_point_given": False,
        "cooldown_counter": 0,
    },
    "forearm": {
        "rep_counter": 0,
        "bonus_given": False,
        "in_motion": False,
        "first_point_given": False,
        "cooldown_counter": 0,
    },
    "pulse": {
        "rep_counter": 0,
        "bonus_given": False,
        "in_motion": False,
        "first_point_given": False,
        "cooldown_counter": 0,
    },
}

COOLDOWN_FRAMES = 5  # frames de segurança


# ==============================
# FUNÇÃO BASE PARA REPETIÇÃO
# ==============================
def process_repetition(movement: str, condition_up: bool) -> int:
    """
    Lógica genérica de detecção de movimentos repetitivos.
    - Ponto inicial na primeira subida
    - Ciclo completo = subida + descida (com cooldown)
    - Ponto extra após 4 ciclos
    """
    state = movement_states[movement]
    score = 0

    # SUBIDA
    if condition_up and not state["in_motion"]:
        state["in_motion"] = True

        if not state["first_point_given"]:
            score += 1
            state["first_point_given"] = True

        state["cooldown_counter"] = 0

    # DESCIDA
    elif not condition_up and state["in_motion"]:
        state["cooldown_counter"] += 1

        if state["cooldown_counter"] >= COOLDOWN_FRAMES:
            state["in_motion"] = False
            state["rep_counter"] += 1
            state["cooldown_counter"] = 0

            if state["rep_counter"] >= 4 and not state["bonus_given"]:
                score += 1
                state["bonus_given"] = True

    # RESET cooldown se parado
    else:
        state["cooldown_counter"] = 0

    return score


# ==============================
# MOVIMENTOS ESPECÍFICOS
# ==============================
def shoulder_abducted(shoulder: float, elbow: float) -> int:
    condition_up = elbow <= shoulder
    return process_repetition("abducted", condition_up)


def shoulder_elevated(shoulder_l: float, shoulder_r: float, side: int = 0,
                      tolerance: float = 30) -> int:
    state = movement_states["elevated"]

    shoulder_y = shoulder_r if side == 0 else shoulder_l
    state["history"].append(shoulder_y)

    if len(state["history"]) > 5:
        state["history"].pop(0)

    if len(state["history"]) < 2:
        return 0

    diff = abs(state["history"][-2] - state["history"][-1])
    condition_up = diff > tolerance

    return process_repetition("elevated", condition_up)


def arm_supported(wrist: float, elbow: float, shoulder: float) -> int: 
    condition_up = (wrist > 450 and elbow > shoulder)
    return process_repetition("supported", condition_up)


def forearm(shoulder_l: float, shoulder_r: float, wrist: float, elbow: float,
            side: int) -> int:
    central_point = (shoulder_l + shoulder_r) / 2

    if side == 0:  # braço direito
        condition_up = (elbow < central_point or wrist < central_point)
    else:  # braço esquerdo
        condition_up = (elbow > central_point or wrist > central_point)

    return process_repetition("forearm", condition_up)


def pulse(
    wrist_x: float,
    wrist_y: float,
    finger_mcp_x: float,
    finger_mcp_y: float,
    pinky_mcp_x: float,
    pinky_mcp_y: float,
    side: int,
) -> int:
    vetor_mcp = np.array([pinky_mcp_x, pinky_mcp_y]) - np.array([finger_mcp_x, finger_mcp_y])
    vetor_palma = np.array([wrist_x, wrist_y]) - np.array([finger_mcp_x, finger_mcp_y])

    eixo_vertical = np.array([0, -1])
    angulo_rad = np.arccos(np.dot(vetor_palma, eixo_vertical) / np.linalg.norm(vetor_palma))
    angulo_flex = abs(np.degrees(angulo_rad) - 160)

    condition_up = angulo_flex > 15  # considera repetição se punho muito fletido/estendido

    return process_repetition("pulse", condition_up)
