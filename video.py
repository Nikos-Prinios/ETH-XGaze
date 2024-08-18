import cv2
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
import dlib
from imutils import face_utils
import os
import csv

# Définition des constantes
VIDEO_FILE_NAME = 'path-to-your-video.mp4'
OUTPUT_FILE_NAME = 'path-to-your-output-video.mp4'
HEATMAP_OUTPUT = 'path-to-your-heatmap.png'

CAMERA_CALIBRATION = 'adjust-the-path-to-your-cam00.xml'
MODEL_PATH = 'adjust-the-path-to-your-epoch_24_ckpt.pth.tar'


# Transformation pour le modèle de gaze
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def normalize_points(points, frame_shape):
    """Normalise les points pour qu'ils occupent tout l'espace de la heatmap."""
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    normalized_points = []
    for point in points:
        x = (point[0] - x_min) / (x_max - x_min) * (frame_shape[1] - 1)
        y = (point[1] - y_min) / (y_max - y_min) * (frame_shape[0] - 1)
        normalized_points.append([x, y])

    return normalized_points

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec


def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    focal_norm = 960
    distance_norm = 600
    roiSize = (224, 224)

    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = np.dot(hR, face_model.T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    z_scale = distance_norm / distance
    cam_norm = np.array([
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
    img_warped = cv2.warpPerspective(img, W, roiSize)

    landmarks_warped = cv2.perspectiveTransform(landmarks.reshape(1, -1, 2), W).reshape(-1, 2)

    return img_warped, landmarks_warped


def process_frame(frame, face_detector, predictor, face_model, camera_matrix, camera_distortion, gaze_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(rgb_frame, 1)
    gaze_points = []

    for face in detected_faces:
        shape = predictor(rgb_frame, face)
        shape = face_utils.shape_to_np(shape)
        landmarks = shape

        facePts = face_model.reshape(6, 1, 3)
        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
        landmarks_sub = landmarks_sub.astype(float).reshape(6, 1, 2)
        hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

        img_normalized, _ = normalizeData_face(frame, face_model, landmarks_sub, hr, ht, camera_matrix)

        input_var = img_normalized[:, :, [2, 1, 0]]
        input_var = trans(input_var)
        input_var = torch.autograd.Variable(input_var.float()).unsqueeze(0)
        pred_gaze = gaze_model(input_var)[0]
        pred_gaze_np = pred_gaze.cpu().data.numpy()

        face_center = np.mean(landmarks_sub, axis=0).flatten().astype(int)
        gaze_length = 300
        gaze_point = face_center + gaze_length * np.array(
            [-np.sin(pred_gaze_np[1]) * np.cos(pred_gaze_np[0]), -np.sin(pred_gaze_np[0])])
        gaze_points.append(gaze_point)

        cv2.arrowedLine(frame, tuple(face_center), tuple(gaze_point.astype(int)), (0, 0, 255), 2)

    return frame, gaze_points


def create_heatmap(gaze_points, shape=(1080, 1920), kernel_size=51):
    """Crée une heatmap à partir des points de regard normalisés."""
    heatmap = np.zeros(shape, dtype=np.float32)
    for point in gaze_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            cv2.circle(heatmap, (x, y), 5, (1, 1, 1), -1)

    # Assurez-vous que kernel_size est impair et supérieur à 1
    kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)

    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def main():
    cap = cv2.VideoCapture(VIDEO_FILE_NAME)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir le fichier vidéo {VIDEO_FILE_NAME}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, fps, (frame_width, frame_height))

    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()

    fs = cv2.FileStorage(CAMERA_CALIBRATION, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('Camera_Matrix').mat()
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()

    face_model_load = np.loadtxt('face_model.txt')
    landmark_use = [20, 23, 26, 29, 15, 19]
    face_model = face_model_load[landmark_use, :]

    gaze_model = gaze_network()
    try:
        ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            gaze_model.load_state_dict(ckpt['model_state'])
        else:
            gaze_model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return
    gaze_model.eval()

    all_gaze_points = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Traiter une image sur 5 pour accélérer
            continue

        processed_frame, gaze_points = process_frame(frame, face_detector, predictor, face_model, camera_matrix,
                                                     camera_distortion, gaze_model)
        all_gaze_points.extend(gaze_points)

        out.write(processed_frame)

        cv2.imshow('Gaze Estimation', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if not all_gaze_points:
        print("Aucun point de regard n'a été détecté.")
        return

    # Normaliser les points de regard
    normalized_points = normalize_points(all_gaze_points, (1080, 1920))

    # Créer et sauvegarder la heatmap
    heatmap = create_heatmap(normalized_points, shape=(1080, 1920))
    cv2.imwrite(HEATMAP_OUTPUT, heatmap)

    # Sauvegarder les points normalisés dans un fichier CSV
    csv_output = '/Users/nikos/Desktop/flux/gaze_points.csv'
    with open(csv_output, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x', 'y'])  # En-tête
        csv_writer.writerows(normalized_points)

    print(f'Vidéo traitée sauvegardée dans : {OUTPUT_FILE_NAME}')
    print(f'Heatmap sauvegardée dans : {HEATMAP_OUTPUT}')
    print(f'Points de regard sauvegardés dans : {csv_output}')


if __name__ == '__main__':
    main()

