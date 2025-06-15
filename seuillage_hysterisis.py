import cv2
import os
import numpy as np

def get_neighboring_frames(filenames, index):
    prev_img = cv2.imread(filenames[index - 1]) if index > 0 else None
    next_img = cv2.imread(filenames[index + 1]) if index < len(filenames) - 1 else None
    return prev_img, next_img

def correct_overexposed_regions(current_img, mask, prev_img, next_img):
    corrected_img = current_img.copy()

    if prev_img is not None and next_img is not None:
        avg_img = cv2.addWeighted(prev_img, 0.5, next_img, 0.5, 0)
    elif prev_img is not None:
        avg_img = prev_img
    elif next_img is not None:
        avg_img = next_img
    else:
        return current_img  # Rien à corriger sans voisins

    for c in range(3):  # Pour chaque canal (BGR)
        corrected_img[:, :, c][mask == 255] = avg_img[:, :, c][mask == 255]

    return corrected_img

def hysteresys_mask(gray, strong_thresh=240, weak_thresh=235):
    # Masque fort (zones très blanches)
    strong_mask = (gray > strong_thresh).astype(np.uint8)

    # Masque faible (zones un peu surexposées)
    weak_mask = (gray > weak_thresh).astype(np.uint8)

    # Étiquetage des composantes connexes sur le masque faible
    num_labels, labels = cv2.connectedComponents(weak_mask)

    # Extraction de la zone forte (zone centrale très blanche)
    strong_dilated = cv2.dilate(strong_mask, np.ones((3, 3), np.uint8), iterations=1)

    # On garde les composantes du masque faible qui touchent la zone forte
    final_mask = np.zeros_like(gray, dtype=np.uint8)
    for label in range(1, num_labels):  # 0 = fond
        component_mask = (labels == label).astype(np.uint8)
        if np.any(strong_dilated & component_mask):
            final_mask = cv2.bitwise_or(final_mask, component_mask)

    return final_mask * 255  # Remise à l'échelle pour affichage/correction

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    filenames = sorted([
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    for i, file_path in enumerate(filenames):
        print(f"🖼️ Traitement : {file_path}")
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 🧠 Génération du masque par hystérésis
        mask = hysteresys_mask(gray, strong_thresh=240, weak_thresh=235)

        # 🔄 Correction par interpolation entre les frames voisines
        prev_img, next_img = get_neighboring_frames(filenames, i)
        corrected_img = correct_overexposed_regions(image, mask, prev_img, next_img)

        # 💾 Enregistrement de l’image corrigée
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, corrected_img)

    print("✅ Toutes les images ont été corrigées et enregistrées.")

# === Utilisation ===
input_folder = "C:/Users/h09184/Downloads/Security-of-a-document-by-a-hologram-TER-py-main/Security-of-a-document-by-a-hologram-TER-py-main/images/origins/passport/psp07_03_03"
output_folder = "C:/Users/h09184/Downloads/Security-of-a-document-by-a-hologram-TER-py-main/Security-of-a-document-by-a-hologram-TER-py-main/images/output3"
process_folder(input_folder, output_folder)
import cv2

image1_path = "C:/Users/h09184/Downloads/Security-of-a-document-by-a-hologram-TER-py-main/Security-of-a-document-by-a-hologram-TER-py-main/images/output_seuillages/psp10_05_03/img_0019.jpg"
image1 = cv2.imread(image1_path)

if image1 is not None:
    print(image1.shape)
else:
    print("Erreur : image non chargée. Vérifie le chemin.")
