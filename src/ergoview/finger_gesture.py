from capture import Landmarks
from utils import calculate_distance


class GestureDetector:
    def __init__(self, landmarks: Landmarks):
        if not hasattr(landmarks, 'side'):
            raise ValueError('Não foram encontrados landmarks das mãos')

        self.landmarks = landmarks
        self.hand_side = landmarks.side
        self.ref_distancia = calculate_distance(
            landmarks[0], landmarks[12]
        )  # tamanho da mão

    def detect_gesture(self) -> str:
        lm = self.landmarks.points

        # Mão aberta -> todos os dedos esticados
        if (
            lm[8].y < lm[6].y
            and lm[12].y < lm[10].y
            and lm[16].y < lm[14].y
            and lm[20].y < lm[18].y
            and (
                (self.hand_side == 'Right' and lm[4].x > lm[3].x)
                or (
                    self.hand_side == 'Left' and lm[4].x < lm[3].x
                )  # Ajuste para a mão esquerda
            )
        ):
            return 'Mao Aberta'

        if (
            calculate_distance(lm[4], lm[8]) < self.ref_distancia * 0.4
            and calculate_distance(lm[4], lm[12]) < self.ref_distancia * 0.4
            and lm[16].y > lm[14].y
            and lm[20].y > lm[18].y
        ):
            return 'Manejo Grosseiro'

        if (
            calculate_distance(lm[4], lm[8]) < self.ref_distancia * 0.3
            and lm[12].y > lm[10].y
            and lm[16].y > lm[14].y
            and lm[20].y > lm[18].y
        ):
            return 'Pinca'

        if (
            lm[8].y < lm[6].y
            and lm[12].y > lm[10].y
            and lm[16].y > lm[14].y
            and lm[20].y > lm[18].y
        ):
            return 'Indicador Apontando'

        if (
            lm[8].y > lm[6].y
            and lm[12].y > lm[10].y
            and lm[16].y > lm[14].y
            and lm[20].y > lm[18].y
        ):
            return 'Punho Fechado'

        return 'Nenhum Gesto Reconhecido'

    def get_hand_info(self) -> str:
        gesture = self.detect_gesture()
        hand_side = 'Esquerda' if self.hand_side == 'Left' else 'Direita'
        return f'Mao: {hand_side} | Gesto: {gesture}'
