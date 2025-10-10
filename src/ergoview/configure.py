import copy

import cv2
import mediapipe as mp

from score_rula import *

score_pulse_r = 0
score_pulse_l = 0

# Inicializa o MP
def initialize_mp_hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mp_draw = mp.solutions.drawing_utils

    return mp_hands, hands, mp_draw


def initialize_mp_pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp_pose, pose


def draw_text(
    frame, text, position, font_scale=0.6, thickness=1, text_color=(0, 255, 0)
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_unified_landmarks(
    frame,
    landmarks,
    connections,
    color_points=(0, 255, 0),
    thickness_points=2,
    circle_radius_points=2,
    color_connections=(0, 255, 0),
    thickness_connections=2,
):
    mp_draw = mp.solutions.drawing_utils
    spec_points = mp_draw.DrawingSpec(
        color=color_points,
        thickness=thickness_points,
        circle_radius=circle_radius_points,
    )
    spec_connections = mp_draw.DrawingSpec(
        color=color_connections,
        thickness=thickness_connections,
        circle_radius=circle_radius_points,
    )
    mp_draw.draw_landmarks(
        frame, landmarks, connections, spec_points, spec_connections
    )


def draw_pose_without_hands(
    frame,
    landmarks,
    mp_pose,
    color_points=(0, 0, 255),
    thickness_points=2,
    circle_radius_points=2,
    color_connections=(255, 255, 255),
    thickness_connections=2,
):
    if not landmarks:
        return

    mp_draw = mp.solutions.drawing_utils

    # copia para esconder os pontos das mãos
    landmarks_copy = copy.deepcopy(landmarks)

    # pontos das mãos (17 a 22)
    for idx in range(17, 23):
        landmarks_copy.landmark[idx].visibility = 0.0

    connections = set(mp_pose.POSE_CONNECTIONS)
    hand_indices = {17, 18, 19, 20, 21, 22}
    filtered_connections = [
        c
        for c in connections
        if c[0] not in hand_indices and c[1] not in hand_indices
    ]

    spec_points = mp_draw.DrawingSpec(
        color=color_points,
        thickness=thickness_points,
        circle_radius=circle_radius_points,
    )
    spec_connections = mp_draw.DrawingSpec(
        color=color_connections, thickness=thickness_connections
    )

    mp_draw.draw_landmarks(
        frame,
        landmarks_copy,
        filtered_connections,
        spec_points,
        spec_connections,
    )

def get_hand_landmark_position_pixels(landmarks, index, frame):
    """
    Retorna a posição em pixels de um landmark específico.

    Parâmetros:
    - landmarks: Lista de landmarks detectados pelo MediaPipe.
    - index: Índice do landmark desejado (ex: 4 para ponta do polegar).
    - frame: Frame de imagem atual (para calcular largura e altura).

    Retorna:
    - (x_pixel, y_pixel): Coordenadas em pixels do landmark.
    """
    h, w = frame.shape[:2]
    x_pixel = int(landmarks.landmark[index].x * w)
    y_pixel = int(landmarks.landmark[index].y * h)
    return x_pixel, y_pixel


def draw_hand_landmarks(
    frame,
    landmarks,
    mp_hands,
    color_points=(0, 0, 255),
    thickness_points=2,
    circle_radius_points=2,
    color_connections=(255, 255, 255),
    thickness_connections=2,
):

    global score_pulse_l
    global score_pulse_r

    idx_finger_mcp, idx_pinky_mcp = 5, 17
    idx_wrist = 0

    wrist_x, wrist_y = get_hand_landmark_position_pixels(landmarks, idx_wrist, frame)
    finger_mcp_x, finger_mcp_y = get_hand_landmark_position_pixels(landmarks, idx_finger_mcp, frame)
    pinky_mcp_x, pinky_mcp_y = get_hand_landmark_position_pixels(landmarks, idx_pinky_mcp, frame)

    score_pulse_r = pulse(wrist_x, wrist_y, finger_mcp_x, finger_mcp_y, pinky_mcp_x, pinky_mcp_y, 0)
    score_pulse_l = pulse(wrist_x, wrist_y, finger_mcp_x, finger_mcp_y, pinky_mcp_x, pinky_mcp_y, 1)


    # Desenha os landmarks na imagem
    draw_unified_landmarks(
        frame,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        color_points,
        thickness_points,
        circle_radius_points,
        color_connections,
        thickness_connections,
    )



def draw_arm_angles(frame, pose_processor, pose_landmarks):
    global rula_left
    global rula_right
    """Calcula e exibe a pontuação RULA para os braços direito e esquerdo,
    alinhando o texto à direita do frame.

    Parâmetros
    ----------
    frame : np.ndarray
        Imagem (frame) onde os textos serão desenhados.
    pose_processor : PoseProcessor
        Processador de pose.
    pose_landmarks : List[Landmark]
        Lista de pontos de referência da pose detectados pelo MediaPipe.
    """
    try:

        """Esse objeto vai ficar com as respostas que preciso colocar no payload"""
        payload_response = [0] * 27

        angle_right, angle_left = pose_processor.get_arm_angles(pose_landmarks)

        #Index dos pontos do corpo
        idx_shoulder_r, idx_elbow_r = 11, 13
        idx_shoulder_l, idx_elbow_l = 12, 14
        idx_wrist_l, idx_wrist_r = 16, 15
        idx_hip_r, idx_hip_l = 23,24

        wrist_l = pose_landmarks[idx_wrist_l]
        wrist_r =  pose_landmarks[idx_wrist_r]
        shoulder_r = pose_landmarks[idx_shoulder_r]
        shoulder_l = pose_landmarks[idx_shoulder_l]
        elbow_l = pose_landmarks[idx_elbow_l]
        elbow_r =  pose_landmarks[idx_elbow_r]
        hip_r = pose_landmarks[idx_hip_r]
        hip_l = pose_landmarks[idx_hip_l]
        
        rula_shoulder_r = shoulder_elevated(shoulder_l.y, shoulder_r.y, 0)
        rula_shoulder_l = shoulder_elevated(shoulder_l.y, shoulder_r.y, 1)
        rula_abducted_l = shoulder_abducted(shoulder_l.y, elbow_l.y)
        rula_abducted_r = shoulder_abducted(shoulder_r.y, elbow_r.y)
        rular_arm_supported = arm_supported(wrist_l.y, elbow_l.y, shoulder_l.y)
        rula_forearm_l = forearm(shoulder_l.x, shoulder_r.x, wrist_l.x, elbow_l.x, 1)
        rula_forearm_l = forearm(shoulder_l.x, shoulder_r.x, wrist_r.x, elbow_r.x, 0)


        #Calculo RULA lado direito
        rula_right = calculate_rula(
            angle_right,
            pose_landmarks[idx_elbow_r].y,
            pose_landmarks[idx_shoulder_r].y,
        ) + score_pulse_r

        #Calculo RULA lado esquerdo
        rula_left = calculate_rula(
            angle_left,
            pose_landmarks[idx_elbow_l].y,
            pose_landmarks[idx_shoulder_l].y,
        ) + score_pulse_l


        #Config Texto
        font_scale = 0.8
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 10
        h, w = frame.shape[:2]

        #Texto de pontuação do RULA
        text_right = f'RULA - Braco Direito: {rula_right}'
        payload_response[12] = rula_right
        (text_width_right, _), _ = cv2.getTextSize(
            text_right, font, font_scale, thickness
        )
        pos_right = (w - text_width_right - margin, 30)
        draw_text(
            frame,
            text_right,
            pos_right,
            font_scale=font_scale,
            thickness=thickness,
            text_color=(0, 255, 0),
        )

        text_left = f'RULA - Braco Esquerdo: {rula_left}'
        payload_response[15] = rula_left
        (text_width_left, _), _ = cv2.getTextSize(
            text_left, font, font_scale, thickness
        )
        pos_left = (w - text_width_left - margin, 60)
        draw_text(
            frame,
            text_left,
            pos_left,
            font_scale=font_scale,
            thickness=thickness,
            text_color=(0, 255, 0),
        )

        
    except Exception as e:
        font_scale = 0.8
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 10
        h, w = frame.shape[:2]
        text_error = f'Erro ao calcular RULA: {e}'
        (text_width_err, _), _ = cv2.getTextSize(
            text_error, font, font_scale, thickness
        )
        pos_err = (w - text_width_err - margin, 90)
        draw_text(
            frame,
            text_error,
            pos_err,
            font_scale=font_scale,
            thickness=thickness,
            text_color=(0, 0, 255),
        )
