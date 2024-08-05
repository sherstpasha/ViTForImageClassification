from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import ViTImageProcessor
import numpy as np
import torch
from torch.nn import Module
from PIL import Image
from typing import Tuple


# Определение трансформаций для изображений
def get_transforms(model_name: str) -> Compose:
    """
    Функция для получения трансформаций, специфичных для модели.

    :param model_name: Название модели, для которой необходимо получить трансформации.
    :return: Объект Compose с необходимыми преобразованиями.
    """
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = image_processor.size['height']
    return Compose([
        Resize((size, size)),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

def predict_image_class_binary(image_array: np.ndarray,
                               model: Module,
                               transforms: Compose,
                               device: torch.device,
                               threshold: float = 0.5) -> Tuple[int, float]:
    """
    Функция для предсказания бинарного класса изображения с использованием заданного порога вероятности.

    :param image_array: Изображение в виде numpy массива (например, загруженное через cv2).
                        Ожидается, что изображение будет в формате HxWxC, где C - количество каналов (обычно 3 для RGB).
    :param model: Модель, которая будет использоваться для предсказания.
                  Ожидается, что модель является экземпляром класса torch.nn.Module.
    :param transforms: Трансформации из torchvision.transforms.Compose, которые будут применяться к изображению.
                       Эти трансформации должны включать преобразование изображения в тензор.
    :param device: Устройство (CPU или GPU), на котором будет выполнено предсказание.
                   Ожидается, что это torch.device объект.
    :param threshold: Порог вероятности для классификации целевого класса.
                      Если вероятность целевого класса выше или равна этому порогу, класс будет считаться "1", иначе "0".
                      По умолчанию установлен на 0.5.
    :return: Кортеж, содержащий предсказанный класс изображения (0 или 1) и вероятность целевого класса.
    """
    # Преобразование numpy массива в PIL Image
    image = Image.fromarray(image_array)

    # Проверка на количество каналов и преобразование к RGB, если нужно
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Применение трансформаций к изображению
    image = transforms(image).unsqueeze(0).to(device)
    
    # Получение предсказания
    with torch.no_grad():
        outputs = model(image).logits
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # Применение порога вероятности для определения класса
    predicted_class = 1 if probs[1] >= threshold else 0
    
    return predicted_class, probs[1]

