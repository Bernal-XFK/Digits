import io
import base64
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

def preprocess_canvas_image(image_input):
    """
    Convierte una imagen de lienzo (PIL Image, numpy array o base64 string)
    al formato exacto requerido por el modelo UCI Digits (8x8 píxeles normalizados a [0, 1]).
    
    Pasos:
    1. Conversión a escala de grises.
    2. Inversión si el fondo es blanco (el modelo espera fondo 0 y trazo > 0).
    3. Detección del bounding box del trazo dibujado.
    4. Recorte y ajuste cuadrático preservando la relación de aspecto con padding.
    5. Centrado por centro de masa (NIST preprocessing standard).
    6. Redimensionamiento a 8x8 con interpolación antialiasing.
    7. Escalado de intensidad a [0.0, 1.0].
    """
    # 1. Cargar imagen
    if isinstance(image_input, str):
        if "," in image_input:
            image_input = image_input.split(",")[1]
        image_bytes = base64.b64decode(image_input)
        img = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise ValueError("Formato de imagen no soportado")

    # Si tiene canal alfa (transparencia), componer sobre fondo blanco o negro
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background

    # 2. Convertir a escala de grises (L)
    img_gray = img.convert('L')
    arr = np.array(img_gray, dtype=np.float32)

    # 3. Detectar si el fondo es blanco y el trazo oscuro
    # Si la media del borde es cercana a 255, el fondo es blanco
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border_pixels.mean() > 127:
        arr = 255.0 - arr  # Invertir: trazo blanco (>0), fondo negro (0)

    # Umbralizar ligeramente para eliminar ruido tenue de fondo
    threshold = 20.0
    arr[arr < threshold] = 0.0

    if arr.max() == 0:
        # Lienzo vacío
        return np.zeros((1, 64), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)

    # 4. Encontrar bounding box de los píxeles con trazo
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Recortar solo el dígito
    cropped = arr[ymin:ymax+1, xmin:xmax+1]
    crop_h, crop_w = cropped.shape

    # Crear caja cuadrada con padding proporcional (20% de margen)
    max_dim = max(crop_h, crop_w)
    pad = int(max_dim * 0.25)
    square_size = max_dim + 2 * pad
    square = np.zeros((square_size, square_size), dtype=np.float32)

    # Posicionar en el centro de la caja cuadrada
    y_offset = (square_size - crop_h) // 2
    x_offset = (square_size - crop_w) // 2
    square[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped

    # 5. Centrar por Centro de Masa (Standard NIST)
    cy, cx = ndimage.center_of_mass(square)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = int(round(square_size / 2.0 - cy))
        shift_x = int(round(square_size / 2.0 - cx))
        square = ndimage.shift(square, [shift_y, shift_x], cval=0.0)

    # 6. Redimensionar a 8x8 con PIL y filtro LANCZOS/BICUBIC
    square_img = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8))
    img_8x8 = square_img.resize((8, 8), Image.Resampling.LANCZOS)
    arr_8x8 = np.array(img_8x8, dtype=np.float32)

    # 7. Escalar intensidades a rango de UCI Digits (0..16) y luego normalizar a [0..1]
    if arr_8x8.max() > 0:
        arr_8x8 = (arr_8x8 / arr_8x8.max()) * 16.0
    arr_8x8 = np.round(arr_8x8)  # Discretizar a enteros 0..16 como UCI Digits
    arr_norm = arr_8x8 / 16.0    # Normalizar para los clasificadores

    return arr_norm.flatten().reshape(1, -1), arr_8x8

if __name__ == "__main__":
    # Test con un dígito simulado
    test_img = np.zeros((200, 200), dtype=np.uint8)
    # Dibujar un trazo vertical simple (dígito 1)
    test_img[30:170, 95:105] = 255
    flat_feat, matrix_8x8 = preprocess_canvas_image(test_img)
    print("Test preprocess successful!")
    print("Shape:", flat_feat.shape)
    print("8x8 Matrix representation:\n", matrix_8x8.astype(int))
