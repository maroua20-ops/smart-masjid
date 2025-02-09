from ultralytics import YOLO
import cv2

# Charger le modèle YOLO pré-entraîné
model = YOLO("yolov8n.pt")  # Modèle léger pour vitesse rapide

# Charger la vidéo
video_path = "mosquee.mp4"  # Mets ici le chemin de ta vidéo
cap = cv2.VideoCapture(video_path)

# Définir une ligne de comptage
line_position = 300  # Position Y de la ligne
entrances = 0
exits = 0
tracked_ids = {}  # Dictionnaire pour suivre les personnes

# Lire la vidéo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fin de la vidéo

    # Appliquer YOLO pour détecter les objets
    results = model(frame)

    # Dessiner la ligne virtuelle
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 2)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées de la boîte
            conf = box.conf[0].item()  # Confiance
            cls = int(box.cls[0].item())  # Classe

            if model.names[cls] == "person":  # Si c'est une personne
                center_y = (y1 + y2) // 2  # Calcul du centre vertical de la boîte
                
                # Détecter entrée ou sortie
                person_id = box.id if hasattr(box, "id") else None  # ID pour tracking
                if person_id not in tracked_ids:
                    tracked_ids[person_id] = center_y
                
                if person_id in tracked_ids:
                    previous_y = tracked_ids[person_id]

                    if previous_y < line_position and center_y >= line_position:
                        entrances += 1  # Personne entre
                    elif previous_y > line_position and center_y <= line_position:
                        exits += 1  # Personne sort

                    tracked_ids[person_id] = center_y  # Mettre à jour la position

                # Dessiner la boîte et afficher le nombre
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher les compteurs
    cv2.putText(frame, f"Entrées: {entrances}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Sorties: {exits}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Afficher la vidéo en temps réel
    cv2.imshow("Détection et Comptage", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
