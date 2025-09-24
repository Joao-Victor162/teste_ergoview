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
from sphinx.builders.latex.nodes import thebibliography

from capture import PoseProcessor, PoseProcessorYOLO
from configure import draw_text
from finger_gesture import GestureDetector
from utils import load_yolo, integration_ml
from PIL import Image
from pypylon import pylon
import pillow_avif

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

def send_payload_to_backend(id_monitoramento, data_list):
    """Monta e envia o payload final para o backend no novo formato."""

    with open("dados_monitoramento.json", "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    try:
        response = requests.post(backend_url, json=data_list, timeout=10)
        response.raise_for_status()
        print(f"Dados enviados. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERRO: Falha ao enviar dados para o backend: {e}")

def teste_duas_cameras(camera, cronometro):

    prev_time = time.time()
    frame_count = 0
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Não foi possível acessar a câmera no índice {camera}")

    while True:
        ret, frame = cap.read()

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time-prev_time
        predicted_class_ml = integration_ml(frame)
        print(f"Classe de machine learning: {predicted_class_ml}")

        if elapsed_time >= 1.0:
            fps = frame_count/elapsed_time
            print(f"FPS DA CAMERA 2:{fps:.2f}")

        cv2.imshow("Teste de cameras", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def video_capture_worker(index_cam, cronometro, id_monitoramento):
    """Worker para capturar frames da câmera e colocá-los na fila."""
    contador = 0

    prev_time = time.time()
    frame_count = 0

    cap = cv2.VideoCapture(index_cam)
    if not cap.isOpened():
        print(f'Não foi possível acessar a câmera no índice {index_cam}.')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    payload_response = [None] * 27

    pose_processor = PoseProcessor(filter_type='ema', alpha=0.3)
    net, classes, output_layers = load_yolo()
    processor = PoseProcessorYOLO(pose_processor, net, classes, output_layers)

    collected_data = []
    total_duration_seconds = cronometro + 2
    total_duration_seconds = cronometro + 2
    process_start_time = time.time()
    timer_payload = 60
    last_capture_time = process_start_time

    print(f"Thread de processamento iniciada. Coletando dados por {total_duration_seconds} segundos.")

    print("Thread de processamento iniciada.")

    while True:
        current_time = time.time()
        current_time_payload = time.time()
        frame_count += 1
        current_time_fps = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time > 1.0:
            fps = frame_count/elapsed_time
            print(f"FPS:{fps:.2f}")
        #print(f'current_time:{current_time - process_start_time}')
        if current_time - process_start_time >= total_duration_seconds:
            if collected_data:
                # print(f"[{datetime.datetime.now()}] Enviando {len(collected_data)} registros para o backend.")
                #send_payload_to_backend(id_monitoramento, collected_data)
                #collected_data.clear()  # Limpa para próxima rodada
                break
            else:
                print("Nenhum dado coletado neste intervalo.")

            process_start_time = current_time

        if current_time_payload - process_start_time >= timer_payload - 1:
            send_payload_to_backend(id_monitoramento, collected_data)
            timer_payload += 60

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        predicted_class_ml = integration_ml(frame=frame)
        # print(f"predicted_class:{predicted_class_ml}")
        paylaod_hand_count = [0] * 3
        """Aqui eu criei um contador para os movimentos das maos"""

        if predicted_class_ml == 0:
            paylaod_hand_count[0] = paylaod_hand_count[0] + 1
        if predicted_class_ml == 1 or predicted_class_ml == 2:
            paylaod_hand_count[1] = paylaod_hand_count[1] + 1
        if predicted_class_ml == 3:
            paylaod_hand_count[2] = paylaod_hand_count[2] + 1


        texts_to_display = []
        processed_frame, hand_landmarks_list, _, payload_response = processor.process_frame(frame)  # AQUI EU VOU TER QUE PEGAR AS INFORMAÇÕES DO RULA PARA INSERIR NO PAYLOAD, NAO ESQUECE
        # print(f"payload_response{payload_response}")
        # print(f"tamanho do payload_response: {len(payload_response)}")

        if hand_landmarks_list:
            start_y, line_spacing = 30, 30
            for hand_landmarks in hand_landmarks_list:
                detector = GestureDetector(hand_landmarks)
                texts_to_display.append(detector.get_hand_info())

        if current_time - last_capture_time >= 1.0:

            directory = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            hash_input = directory.encode('utf-8')
            hash_result = hashlib.sha256(hash_input).hexdigest()[:8]
            object_name = f"{id_monitoramento}/frame_{directory}_{hash_result}.avif"
            #print("enviando frames para o minio")
            upload_frame_to_minio(bucketName, object_name, frame)
            contador += 1

            #print(f"Coletando registro {len(collected_data) + 1}/{total_duration_seconds}...")

            data_point = create_data_point(payload_response, paylaod_hand_count, id_monitoramento, object_name)
            paylaod_hand_count[0] = 0
            paylaod_hand_count[0] = 1
            paylaod_hand_count[0] = 2
            collected_data.append(data_point)
            last_capture_time = current_time

            for i, text in enumerate(texts_to_display):
                draw_text(processed_frame, text, (10, start_y + (i * line_spacing)))

        cv2.imshow('Deteccao de Movimentos', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Thread de processamento finalizada.")

    cap.release()
    print("Thread de captura finalizada")
    return contador

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

def using_industrial_cam():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    prev_time = time.time()
    frame_count = 0

    if not camera.IsGrabbing():
        print("Câmera não está capturando")
        exit()

    cv2.namedWindow("Basler Camera Feed", cv2.WINDOW_NORMAL)

    while True:
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

        if grab_result.GrabSucceeded():
            frame = grab_result.Array

            cv2.imshow("Basler Camera Feed", frame)

        grab_result.Release()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.StopGrabbing()
    cv2.destroyAllWindows()


def upload_frame_to_minio(bucket_name, object_name, frame):
    try:
        #print("Enviando frame para o MinIO...")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        buffer = io.BytesIO()
        image.save(buffer, format="AVIF")
        encoded_image = buffer.getvalue()

        minio_client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(encoded_image),
            length=len(encoded_image),
            content_type="image/avif"
        )

        #print(f"Frame enviado para o MinIO como {object_name}")
    except Exception as e:
        print(f"Erro ao enviar o frame para o MinIO: {e}")

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

        if cronometro < 60 or cronometro > 300:
            return jsonify({
                "status":"erro",
                "message":"Tempo de captura não suportado. O valor mínimo é 60 segundos e o máximo é 500 segundos."
            }), 400

        if index_cam == 7:
            thread = threading.Thread(target=video_capture_worker, args=(0, cronometro, id_monitoramento))
            thread_2 = threading.Thread(target=teste_duas_cameras, args=(2, cronometro))

            thread.start()
            thread_2.start()
            return jsonify({
                "status":"sucess",
                "message":"Captura com duas cãmeras iniciadas"
            })

        if index_cam == 3:
            thread = threading.Thread(target=teste_duas_cameras, args=(2, time))
            thread.start()

            return jsonify({
                "status":"sucess",
                "message":"Captura de camera industrial iniciada"
            })

        else:
            #contador = video_capture_worker(index_cam, cronometro, id_monitoramento)
            thread = threading.Thread(target=video_capture_worker, args=(index_cam, cronometro, id_monitoramento))

            thread.start()
            #thread.join()

            return jsonify({
                "status":"sucesso",
                #"count_frame":contador
                "message":f"captura iniciada com a camera {index_cam}"
            })

    except Exception as e:
        return jsonify({"erro": e }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
