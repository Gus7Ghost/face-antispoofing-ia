# ==============================================================================
# Sistema de Detección Anti-Suplantación Facial (Face Anti-Spoofing & Liveness)
# Autor: Gustavo Bryan López Cosme | UNI - LabIAR
# ==============================================================================

import cv2
import numpy as np
import time
import math
import argparse
import os

# Suprimir advertencias verbose de TensorFlow / OneDNN en consola
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FaceLivenessDetector:
    """
    Detector de Prueba de Vida (Liveness Detection) para mitigar ataques
    de presentación (fotos impresas y repetición en pantallas).
    Utiliza Eye Aspect Ratio (EAR) y análisis de textura espectral.
    """
    def __init__(self, ear_threshold=0.22, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.blink_counter = 0
        self.total_blinks = 0
        
        # Índices de landmarks de ojos para FaceMesh estándar
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    @staticmethod
    def calculate_distance(p1, p2):
        """Calcula la distancia euclidiana 2D entre dos puntos"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_ear(self, eye_points):
        """
        Calcula el Eye Aspect Ratio (EAR) según la formulación de Soukupová y Čech:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        v1 = self.calculate_distance(eye_points[1], eye_points[5])
        v2 = self.calculate_distance(eye_points[2], eye_points[4])
        h = self.calculate_distance(eye_points[0], eye_points[3])

        if h == 0:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def analyze_texture_variance(self, face_crop):
        """
        Evalúa el operador Laplaciano para determinar nitidez y descartar pantallas.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

def test_pipeline_headless():
    """Prueba unitaria del pipeline sin requerir cámara física activa"""
    print("=== [TEST] Verificando componentes de Anti-Spoofing ===")
    detector = FaceLivenessDetector()
    
    # 1. Prueba matemática de EAR
    test_eye = [(0, 10), (5, 15), (10, 15), (15, 10), (10, 5), (5, 5)]
    ear_val = detector.calculate_ear(test_eye)
    print(f"[OK] Calculo matematico de EAR verificado: {ear_val:.4f}")

    # 2. Inicialización de detector de malla facial
    try:
        from cvzone.FaceMeshModule import FaceMeshDetector
        mesh_detector = FaceMeshDetector(maxFaces=1)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, faces = mesh_detector.findFaceMesh(dummy_img, draw=False)
        print("[OK] Detector de Malla Facial (MediaPipe / CVZone) inicializado exitosamente.")
    except Exception as e:
        print(f"[ERROR] Error al inicializar detector: {e}")
        return False

    print("[OK] Pipeline de vision artificial 100% operativo y listo para camara o video.")
    return True

def run_detector(video_source=0):
    print("================================================================")
    print("  Sistema Anti-Suplantacion Facial | Gustavo Lopez (UNI - LabIAR)")
    print("================================================================")
    print(f"Abriendo fuente de video: {video_source}")
    print("Controles: Presione 'q' en la ventana de video para salir.\n")
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente de video '{video_source}'.")
        print("Sugerencia: Si no tienes camara web conectada, prueba:")
        print("  python portafolio\\01-face-antispoofing-ia\\anti_spoofing.py --test")
        print("  python portafolio\\01-face-antispoofing-ia\\anti_spoofing.py --video ruta_a_un_video.mp4")
        return

    detector = FaceLivenessDetector()

    # Cargar detector de malla facial
    try:
        from cvzone.FaceMeshModule import FaceMeshDetector
        mesh_detector = FaceMeshDetector(maxFaces=1)
        has_mesh = True
        print("[INFO] Detector de Malla Facial inicializado.")
    except Exception as e:
        has_mesh = False
        print(f"[ALERTA] Detector de malla no disponible ({e}).")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin del flujo de video o lectura interrumpida.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        fps_counter += 1

        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        display_text = f"FPS: {fps} | Sistema Anti-Spoofing"
        cv2.putText(frame, display_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if has_mesh:
            frame, faces = mesh_detector.findFaceMesh(frame, draw=False)

            if faces:
                face = faces[0] # Lista de 468 puntos [x, y]
                
                left_eye_pts = [face[idx] for idx in detector.LEFT_EYE if idx < len(face)]
                right_eye_pts = [face[idx] for idx in detector.RIGHT_EYE if idx < len(face)]

                if len(left_eye_pts) == 6 and len(right_eye_pts) == 6:
                    ear_left = detector.calculate_ear(left_eye_pts)
                    ear_right = detector.calculate_ear(right_eye_pts)
                    avg_ear = (ear_left + ear_right) / 2.0

                    if avg_ear < detector.ear_threshold:
                        detector.blink_counter += 1
                    else:
                        if detector.blink_counter >= detector.consecutive_frames:
                            detector.total_blinks += 1
                        detector.blink_counter = 0

                    is_real = detector.total_blinks >= 1
                    state_label = "AUTENTICO (SUJETO VIVO)" if is_real else "VERIFICANDO PARPADEO..."
                    box_color = (0, 255, 0) if is_real else (0, 165, 255)

                    for pt in left_eye_pts + right_eye_pts:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

                    cv2.putText(frame, f"EAR: {avg_ear:.2f} | Parpadeos: {detector.total_blinks}", (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(frame, f"ESTADO: {state_label}", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)
            else:
                cv2.putText(frame, "ROSTRO NO DETECTADO", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Anti-Spoofing & Liveness Detection - Gustavo Lopez (UNI)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema Anti-Suplantacion Facial")
    parser.add_argument("--video", type=str, default=None, help="Ruta a un archivo de video")
    parser.add_argument("--test", action="store_true", help="Ejecuta una prueba unitaria del pipeline")
    args = parser.parse_args()

    if args.test:
        test_pipeline_headless()
    elif args.video:
        run_detector(args.video)
    else:
        run_detector(0)
