from minio import Minio
import threading
import hashlib
import io
import datetime
import time
import requests
import  json
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from capture import PoseProcessor, PoseProcessorYOLO
from configure import draw_text
from finger_gesture import GestureDetector
from utils import load_yolo, integration_ml
from PIL import Image
import pillow_avif
import queue

stack_frames_for_rula = queue.Queue(maxsize=500)
stack_frames_for_ml = queue.Queue(maxsize=500)
stack_frames_for_avif = queue.Queue(maxsize=500)
stack_frames_for_avif_hands = queue.Queue(maxsize=500)
stack_frames_rula = queue.Queue(maxsize=500)
stack_frames_ml = queue.Queue(maxsize=500)
stack_frames_hall = queue.Queue(maxsize=500)
stack_frames_avif = queue.Queue(maxsize=500)
stacks_ids_minio = queue.Queue(maxsize=500)
stacks_ids_minio_hands = queue.Queue(maxsize=500)
stack_payload_template = queue.Queue(maxsize=500)
stop_event = threading.Event()

global collected_data

app = Flask(__name__)
CORS(app)

"""Carregando as configurações do sistema"""
def load_appsettings(file_path):
    with open(file_path, 'r') as file:
        settings = json.load(file)
    return settings

configs = load_appsettings('appsettings.json')

minio_client = Minio(
            endpoint=configs["Minio"]["baseUrl"],
            access_key=configs["Minio"]["acess_key"],
            secret_key=configs["Minio"]["secret_key"],
            secure=False
        )

bucketName = configs["Minio"]["bucket_name"]
backend_url = configs["Backend"]["url"]

def create_data_point(payload_response, payload_hand_count, id_monitoramento, objectKey):
    return {
        "id_monitoramento": str(id_monitoramento),
        "data": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "posicionamento_ombro_direito": str(payload_response[0]),
            "ombro_abduzido_direito": str(payload_response[1]),
            "ombro_elevado_direito": str(payload_response[2]),
            "braco_apoiado_direito": str(payload_response[3]),
            "score_ombro_direito": str(payload_response[4]),
            "posicionamento_ombro_esquerdo": str(payload_response[5]),
            "ombro_abduzido_esquerdo": str(payload_response[6]),
            "ombro_elevado_esquerdo": str(payload_response[7]),
            "braco_apoiado_esquerdo": str(payload_response[8]),
            "score_ombro_esquerdo": str(payload_response[9]),
            "posicionamento_antebraco_direito": str(payload_response[10]),
            "cruza_linha_media_direito": str(payload_response[11]),
            "score_antebraco_direito": str(payload_response[12]),
            "posicionamento_antebraco_esquerdo": str(payload_response[13]),
            "cruza_linha_media_esquerdo": str(payload_response[14]),
            "score_antebraco_esquerdo": str(payload_response[15]),
            "posicionamento_punho_direito": str(payload_response[16]),
            "desvio_punho_direito": str(payload_response[17]),
            "rotacao_punho_direito": str(payload_response[18]),
            "score_punho_direito": str(payload_response[19]),
            "posicionamento_punho_esquerdo": str(payload_response[20]),
            "desvio_punho_esquerdo": str(payload_response[21]),
            "rotacao_punho_esquerdo": str(payload_response[22]),
            "score_punho_esquerdo": str(payload_response[23]),
            "postura_superior": str(payload_response[24]),
            "rula_score_final_direito": str(payload_response[25]),
            "rula_score_final_esquerdo": str(payload_response[26]),
            "frame_rula": objectKey,
            "movimentos_maos": {
                "manejo_grosseiro": str(payload_hand_count[1]),
                "frame_manejo_grosseiro": objectKey,
                "pinca": str(payload_hand_count[2]),
                "frame_pinca": objectKey,
                "mao_aberta": str(payload_hand_count[0]),
                "frame_mao_aberta": objectKey
                #campo esforço: somar movimentos total + campo peso de acordo com o movimento
                #campo peso: 0 - mao aberta; 5 pinça, 8 - manejo grosseiro
            }
        }
    }

def send_payload_to_backend(data_list):
    """Monta e envia o payload final para o backend no novo formato."""

    with open("dados_monitoramento.json", "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    try:
        #response = requests.post(backend_url, json=data_list, timeout=10)
        #response.raise_for_status()
        #print(f"Dados enviados. Status: {response.status_code}")
        print(f"Dados enviados. Status")
    except requests.exceptions.RequestException as e:
        print(f"ERRO: Falha ao enviar dados para o backend: {e}")

"""
AQUI É O METODO DO HALL QUE EU CRIEI
def count_hall_method(count_moviments):
    result = (count_moviments*100)/60

    if 0 <= result < 20:
        hall_effort = 0
    elif 20 <= result < 40:
        hall_effort = 2
    elif 40 <= result < 60:
        hall_effort = 4
    elif 60 <= result < 80:
        hall_effort = 6
    elif 80 <= result < 100:
        hall_effort = 8
    else:
        hall_effort = 10

    return hall_effort
"""

def calibration_cam(index_cam):
    cap_calibration = None
    try:
        count_frames = 0
        if index_cam == 7:
            cap_calibration = cv2.VideoCapture(0, cv2.CAP_V4L2)
        else:
            cap_calibration = cv2.VideoCapture(index_cam, cv2.CAP_V4L2)

        cap_calibration.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_calibration.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_calibration.set(cv2.CAP_PROP_FPS, 30)
        cap_calibration.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        pose_processor = PoseProcessor(filter_type='ema', alpha=0.3)
        net, classes, output_layers = load_yolo()
        processor = PoseProcessorYOLO(pose_processor, net, classes, output_layers)

        while True:
            ret, frame = cap_calibration.read()
            if not ret:
                print(f'Falha ao ler o frame da câmera no índice {index_cam}.')
                cap_calibration.release()
                return None

            frame = cv2.flip(frame, 1)
            count_frames += 1

            if count_frames == 5:
                if index_cam == 0:
                    _, _, _, _, ps, _ = processor.process_frame(frame)
                    cap_calibration.release()
                    if ps is None:
                        return None
                    return ps

                elif index_cam >= 1 and index_cam != 7:
                    _, _, _, _, _, hs = processor.process_frame(frame)
                    cap_calibration.release()
                    if hs is None:
                        return None
                    return hs

                elif index_cam == 7:
                    _, _, _, _, ps, hs = processor.process_frame(frame)
                    cap_calibration.release()
                    if ps is None or hs is None:
                        return None
                    return ps, hs

    except Exception as e:
        print(f'Erro na calibração: {str(e)}')
        cap_calibration.release()
        return None



def create_stack_frames(index_cam: int, limit_frames: int):
    cap = cv2.VideoCapture(index_cam, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    count_frames = 0
    controller_time = time.time()
    control = 1.0
    print("iniciando o processo de captura de frames")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not cap.isOpened():
            print(f'Não foi possível acessar a câmera no índice {index_cam}.')
            return

        try:
            stack_frames_for_rula.put(frame)
            #stack_frames_for_ml.put(frame)
            stack_frames_for_avif.put(frame)
            count_frames += 1
            print(f"Frames: {count_frames}")
            controller_time += control
            nex_timer = controller_time - time.time()

            if nex_timer > 0:
                time.sleep(nex_timer)

            else:
                controller_time = time.time()

            if count_frames == limit_frames:
                print(f"Quantidade de frames totais capturados.")
                stack_frames_for_rula.put(None)
                #stack_frames_for_ml.put(None)
                stack_frames_for_avif.put(None)
                cap.release()
                break

        except queue.Empty:
            continue

def create_stack_frames_for_ml(index_cam: int, limit_frames: int):
    cap = cv2.VideoCapture(index_cam, cv2.CAP_V4L2)#apagar esse cara quando estiver no linux

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    count_frames = 0
    controller_time = time.time()
    control = 1.0
    print("iniciando o processo de captura de frames")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not cap.isOpened():
            print(f'Não foi possível acessar a câmera no índice {index_cam}.')
            return

        try:
            #stack_frames_for_rula.put(frame)
            stack_frames_for_ml.put(frame)
            stack_frames_for_avif_hands.put(frame)
            count_frames += 1
            print(f"Frames: {count_frames}")
            controller_time += control
            nex_timer = controller_time - time.time()

            if nex_timer > 0:
                time.sleep(nex_timer)

            else:
                controller_time = time.time()

            if count_frames == limit_frames:
                print(f"Quantidade de frames totais capturados.")
                #stack_frames_for_rula.put(None)
                stack_frames_for_ml.put(None)
                stack_frames_for_avif_hands.put(None)
                cap.release()
                break

        except queue.Empty:
            continue

def consumer_and_apply_rula_processing():
    pose_processor = PoseProcessor(filter_type='ema', alpha=0.3)
    net, classes, output_layers = load_yolo()
    processor = PoseProcessorYOLO(pose_processor, net, classes, output_layers)
    print("Iniciando o processo de consumo dos frames")

    while True:
        frames = stack_frames_for_rula.get()
        try:
            if frames is None:
                stack_frames_rula.put(None)
                stack_frames_for_rula.task_done()
                break

            texts_to_display = []
            processed_frame, hand_landmarks_list, _, payload_response, ps = processor.process_frame(frames)
            print(f"handmark_list: {hand_landmarks_list}")

            if hand_landmarks_list:
                start_y, line_spacing = 30, 30
                for hand_landmarks in hand_landmarks_list:
                    detector = GestureDetector(hand_landmarks)
                    texts_to_display.append(detector.get_hand_info())
                    for i, text in enumerate(texts_to_display):
                        draw_text(processed_frame, text, (10, start_y + (i * line_spacing)))

                #cv2.imshow('Deteccao de Movimentos', processed_frame)
                #stack_frames_rula.put(processed_frame)
                stack_frames_rula.put(payload_response)

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break
            #cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error: {e}")

        #finally:
            #stack_frames.task_done()

def consumer_integration_ml():
    contador_ml = 0
    while True:
        frames_for_ml = stack_frames_for_ml.get()
        try:
            if frames_for_ml is None:
                stack_frames_ml.put(None)
                stack_frames_for_ml.task_done()
                break

            predicted_class_ml = integration_ml(frame=frames_for_ml)
            contador_ml += 1
            print(f"imagem processada pelo modelo de ML: {contador_ml}")
            paylaod_hand_count = [0] * 3
            """Aqui eu criei um contador para os movimentos das maos"""

            if predicted_class_ml == 0:
                paylaod_hand_count[0] = paylaod_hand_count[0] + 1
            if predicted_class_ml == 1 or predicted_class_ml == 2:
                paylaod_hand_count[1] = paylaod_hand_count[1] + 1
            if predicted_class_ml == 3:
                paylaod_hand_count[2] = paylaod_hand_count[2] + 1

            stack_frames_ml.put(paylaod_hand_count)

        except Exception as e:
            print(f"Error: {e}")


def consumer_and_convert_for_avif():
    contador = 0
    while True:
        frames_rula = stack_frames_for_avif.get()
        frames_ml = stack_frames_for_ml.get()
        try:

            if frames_rula is None and frames_ml is None:
                stack_frames_avif.put(None)
                stack_frames_for_avif_hands.put(None)
                stack_frames_rula.task_done()
                stack_frames_for_ml.task_done()
                break

            frame_rgb = cv2.cvtColor(frames_rula, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            buffer = io.BytesIO()
            image.save(buffer, format="AVIF")
            encoded_image = buffer.getvalue()
            stack_frames_avif.put(encoded_image)
            contador += 1
            print(f"Enviando encoded image para a pilha do avif: {contador}")

            #para as mãos, depois vou refatorar aqui para ficar coerente, por hora, apenas para validação irei deixar nesse formato repetitivo.
            frame_rgb_hands = cv2.cvtColor(frames_ml, cv2.COLOR_BGR2RGB)
            image_hands = Image.fromarray(frame_rgb_hands)
            buffer_hands = io.BytesIO
            image_hands.save(buffer_hands, format="AVIF")
            encoded_image_hands = buffer_hands.getvalue()
            stack_frames_for_avif_hands.put(encoded_image_hands)

        except Exception as e:
            print(f"Error: {e}")

def upload_frame_for_minio(id_monitoramento, bucket_name):
    contador_minio_frames = 0
    while True:
        frames_avif_bytes = stack_frames_avif.get()
        frames_avif_bytes_hands = stack_frames_for_avif_hands.get()
        try:
            if frames_avif_bytes is None and frames_avif_bytes_hands is None:
                stacks_ids_minio.put(None)
                stacks_ids_minio_hands.put(None)
                stack_frames_avif.task_done()
                stack_frames_for_avif_hands.task_done()

            directory = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            hash_input = directory.encode('utf-8')
            hash_result = hashlib.sha256(hash_input).hexdigest()[:8]
            object_name = f"{id_monitoramento}/rula/frame_{directory}_{hash_result}.avif"
            print(f"object_name: {object_name}")


            #Aqui eh o mesmo processo, apenas para fins de teste irei deixar desoeganizado nessa parte, irei refatorar para nao ficar repetindo código
            directory_hands = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            hash_input_hands = directory_hands.encode('utf-8')
            hash_result_hands = hashlib.sha256(hash_input_hands).hexdigest()[:8]
            object_name_hands =  f"{id_monitoramento}/hall/frame_{directory_hands}_{hash_result_hands}.avif"

            minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=io.BytesIO(frames_avif_bytes),
                length=len(frames_avif_bytes),
                content_type="image/avif"
            )

            minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name_hands,
                data=io.BytesIO(frames_avif_bytes_hands),
                length=len(frames_avif_bytes_hands),
                content_type="image/avif"
            )

            contador_minio_frames += 1
            stacks_ids_minio.put(object_name)
            stacks_ids_minio_hands.put(object_name_hands)
            print(f"Frame enviado para o minio: {contador_minio_frames}")

        except Exception as e:
            print(f"Erro ao enviar o frame para o MinIO: {e}")

def generate_template_payload(id_monitoramento):
    global collected_data

    cronometro_paylaod = time.time()
    collected_data = []
    contador_payload = 0

    while True:
        payload_response_template = stack_frames_rula.get()
        payload_hand_count_template = stack_frames_ml.get()
        ids_minio = stacks_ids_minio.get()
        cronometro_paylaod_resolve = time.time()

        try:
            if payload_response_template is None:
                stack_payload_template.put(None)
                stack_frames_rula.task_done()
                stacks_ids_minio.task_done()
                stack_frames_ml.task_done()
                send_payload_to_backend(collected_data)
                break

            contador_payload += 1
            data_point = create_data_point(payload_response_template, payload_hand_count_template, id_monitoramento, ids_minio)
            #print(f"data_point gerado: {data_point}")
            print(f"Data_point gerado: {contador_payload}")
            collected_data.append(data_point)

        except Exception as e:
            print(f"Error: {e}")


@app.route("/api/start_record", methods=["POST"])
def start_record():
    try:
        dados_payload = request.get_json()
        if not dados_payload:
            return jsonify({
                "status":"erro",
                "message":"Corpo da requisição JSON ausente",
            }), 400

        id_monitoramento = dados_payload.get('id_monitoramento')
        index_cam = int(dados_payload.get('index_cam'))
        cronometro = int(dados_payload.get('time'))

        #if cronometro < 60 or cronometro > 300:
        #    return jsonify({
        #        "status":"erro",
        #        "message":"Tempo de captura não suportado. O valor mínimo é 60 segundos e o máximo é 500 segundos."
        #    }), 400

        if index_cam == 7:
            thread_processing_two_cams = threading.Thread(target=create_stack_frames, args=(0, cronometro, ), name="Thread de montagem de pilha de frames com duas câmeras para o rula")
            thread_processing_two_cams_ml = threading.Thread(target=create_stack_frames_for_ml, args=(1, cronometro,),
                                                          name="Thread de montagem de pilha de frames com duas câmeras para o modelo")
            thread_processing_two_cams.start()
            thread_processing_two_cams_ml.start()
            """
            return jsonify({
                "status":"sucess",
                "message":"Captura com duas cãmeras iniciadas"
            })
            """

        else:
            thread_mouting_stack_frames = threading.Thread(target=create_stack_frames, args=(index_cam, cronometro, True), name="Montagem da pilha de frames")
            thread_mouting_stack_frames.start()

        thread_apply_rula = threading.Thread(target=consumer_and_apply_rula_processing, name="aplicação do rula na primeira pilha de frames")
        thread_apply_ml = threading.Thread(target=consumer_integration_ml, name="Integracao com o modelo de ML")
        thread_convert_for_avif = threading.Thread(target=consumer_and_convert_for_avif, name="Conversao dos frames para o formato avif")
        thread_upload_images_minio = threading.Thread(target=upload_frame_for_minio, args=(id_monitoramento, bucketName), name="Upload dos frames no formato avif para o minio")
        thread_template_payload = threading.Thread(target=generate_template_payload, args=(id_monitoramento), name="Geracao do template do payload no formato de dicionario")


        thread_apply_rula.start()
        thread_apply_ml.start()
        thread_convert_for_avif.start()
        thread_upload_images_minio.start()
        thread_template_payload.start()

        return jsonify({
            "status":"sucesso",
            #"count_frame":contador
            "message":f"captura iniciada com a camera {index_cam}"
        })

    except Exception as e:
        return jsonify({"erro": e }), 500

@app.route("/api/calibration", methods=["POST"])
def calibration():
    try:

        dados_payload = request.get_json()
        if not dados_payload:
            return jsonify({
                "status": "erro",
                "message": "Corpo da requisição JSON ausente",
            }), 400

        index_cam = int(dados_payload.get('index_cam'))
        if index_cam == 0:
            pose_points = calibration_cam(index_cam)
            return jsonify({"message":"calibration complete", "pose_points":pose_points})
        if index_cam >= 1 and index_cam != 7:
            hand_points = calibration_cam(index_cam)
            return jsonify({"message":"calibration complete", "hand_points":hand_points})
        if index_cam == 7:
            pose_points, hand_points = calibration_cam(index_cam)
            return jsonify({"message":"calibration complete", "pose_points":pose_points, "hand_points":hand_points})

    except Exception as e:
        return jsonify({"error":e})

@app.route("/api/get_list_cameras", methods=["GET"])
def get_list_cams():
    list_of_cams = []
    try:
        for index in range(7):
            cap_list_cam = cv2.VideoCapture(index)
            if cap_list_cam.isOpened():
                list_of_cams.append(index)
                print(f"Câmera encontrada no índice: {index}")
                cap_list_cam.release()
            else:
                print(f"Nenhuma câmera encontrada no índice: {index}")
        return jsonify({"data":list_of_cams})
    except Exception as e:
        return jsonify({"error":e})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
