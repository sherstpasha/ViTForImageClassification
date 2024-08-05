import os
from pathlib import Path
from utils import predict_image_class_binary, get_transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from transformers import ViTForImageClassification
import torch
import shutil
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

def classify_images_in_directory(input_dir, output_dir, model, transforms, device, threshold=0.5):
    """
    Классификация всех изображений в указанной директории и сохранение результатов в соответствующие папки.
    
    :param input_dir: Директория, в которой нужно искать изображения.
    :param output_dir: Директория, в которой будут сохранены результаты.
    :param model: Модель для классификации изображений.
    :param transforms: Трансформации для изображений.
    :param device: Устройство для выполнения предсказаний (CPU или CUDA).
    :param threshold: Порог вероятности для классификации целевого класса.
    """
    # Создаем папки для хранения результатов
    class_0_dir = Path(output_dir) / "class_0"
    class_1_dir = Path(output_dir) / "class_1"
    error_dir = Path(output_dir) / "error"

    class_0_dir.mkdir(parents=True, exist_ok=True)
    class_1_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    # Рекурсивный поиск всех изображений в указанной директории
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            image_files.append(file_path)

    # Обработка изображений с немедленной записью результатов
    for file_path in tqdm(image_files, desc="Классификация изображений"):
        try:
            # Загрузка изображения и предсказание класса и вероятности
            image = np.array(Image.open(file_path))
            predicted_class, probability = predict_image_class_binary(
                image_array=image,
                model=model,
                transforms=transforms,
                device=device,
                threshold=threshold
            )
            
            # Формирование имени файла с вероятностью в имени
            output_filename = f"{probability:.4f}_{file_path.name}"

            # Копирование файла в соответствующую папку в зависимости от предсказанного класса
            if predicted_class == 0:
                shutil.copy(file_path, class_0_dir / output_filename)
            elif predicted_class == 1:
                shutil.copy(file_path, class_1_dir / output_filename)

        except (OSError, UnidentifiedImageError):
            # Если изображение не удалось обработать, перемещаем его в папку ошибок
            shutil.copy(file_path, error_dir / file_path.name)

    print(f"Классификация завершена. Результаты сохранены в: {output_dir}")

# Основной блок скрипта
if __name__ == "__main__":
    input_directory = r"C:\content\222"  # Путь к директории с изображениями
    output_directory = r"C:\content\filtered"  # Путь к директории для сохранения результатов

    # Устройство для выполнения предсказания (CUDA, если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели ViT
    model = ViTForImageClassification.from_pretrained(r"C:\Users\user\Desktop\content_filter\ViTMVDModel")
    model.to(device)
    model.eval()

    # Получение трансформаций для модели ViT
    transforms = get_transforms(model_name='google/vit-base-patch16-224')

    # Запуск классификации
    classify_images_in_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        model=model,
        transforms=transforms,
        device=device,
        threshold=0.05
    )
