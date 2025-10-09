import math
from typing import Optional, Tuple
import tensorflow as tf
import numpy as np

import cv2


def calculate_distance(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def is_like(landmarks, frame_width, frame_height):
    """## _summary_

    ### Args:
        - `landmarks (_type_)`: _description_
        - `frame_width (_type_)`: _description_
        - `frame_height (_type_)`: _description_

    ### Returns:
        - `_type_`: _description_
    """
    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    index_tip = landmarks[8]

    thumb_tip_pos = (
        int(thumb_tip.x * frame_width),
        int(thumb_tip.y * frame_height),
    )
    thumb_base_pos = (
        int(thumb_base.x * frame_width),
        int(thumb_base.y * frame_height),
    )
    index_tip_pos = (
        int(index_tip.x * frame_width),
        int(index_tip.y * frame_height),
    )

    thumb_up = (
        thumb_tip_pos[1] < thumb_base_pos[1]
        and calculate_distance(thumb_tip_pos, index_tip_pos) > 50
    )
    return thumb_up


def display_feedback(frame, thumb_status, hand_detected):
    """## _summary_

    ### Args:
        - `frame (_type_)`: _description_
        - `thumb_status (_type_)`: _description_
        - `hand_detected (_type_)`: _description_
    """
    if hand_detected:
        print('Mão detectada!')
    else:
        print('Nenhuma mão detectada.')

    color = (0, 255, 0) if thumb_status == 'Gostei!' else (0, 0, 255)

    cv2.putText(
        frame, thumb_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
    )
    cv2.imshow('Movimento Detectado com as maos', frame)


def calculate_angle(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> Optional[float]:
    """
    Calcula o ângulo entre os vetores formados pelos pontos a, b e c.

    Parâmetros
    ----------
    a: tuple[float, float]
        Coordenadas (x,y) do primeiro ponto (ex: ombro)
    b: tuple[float, float]
        Coordenadas (x,y) do segundo ponto (ex: cotovelo)
    c: tuple[float, float]
        Coordenadas (x,y) do terceiro ponto (ex: pulso)

    Retorna
    -------
    float ou None
        O ângulo em graus entre os vetores AB e BC. Retorna None se a magnitude de algum vetor for zero para evitar divisão por zero.
    """
    ab = (a[0] - b[0], a[1] - b[1])  # Vetor AB
    bc = (c[0] - b[0], c[1] - b[1])  # Vetor BC

    dot_product = ab[0] * bc[0] + ab[1] * bc[1]  # Produto escalar
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)  # Magnitude AB
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)  # Magnitude BC

    if mag_ab == 0 or mag_bc == 0:
        return None  # Evita divisão por zero

    angle = math.acos(dot_product / (mag_ab * mag_bc))  # Ângulo em radianos
    return math.degrees(angle)  # Converte para graus


def load_yolo(
    cfg_path="../yolo/yolov4-tiny.cfg",
    weights_path="../yolo/yolov4-tiny.weights",
    names_path="../yolo/coco.names",
):
    """Carrega o modelo e os arquivos necessários

    Parameters
    ----------
    cfg_path : str, optional
        _description_, by default "yolov4-tiny.cfg"
    weights_path : str, optional
        _description_, by default "yolov4-tiny.weights"
    names_path : str, optional
        _description_, by default "coco.names"

    Returns
    -------
    _type_
        _description_
    """
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def integration_ml(
    frame,
    tflite_path = "../ml_ergoview/modelo_intelbras.tflite",
    target_size=(50, 50)
):

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    normalized = equalized.astype(np.float32) / 255.0
    resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(resized, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])

    predicted_class = np.argmax(output_data)

    return int(predicted_class)
