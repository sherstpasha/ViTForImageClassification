from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import ViTImageProcessor
import numpy as np
import torch
from PIL import Image


# Определение трансформаций для изображений
def get_transforms(model_name: str):
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
                                model: torch.nn.Module,
                                  transforms: Compose,
                                  device,
                                    threshold: float = 0.5):
    """
    Функция для предсказания класса изображения с учетом порога вероятности.
    
    :param image_array: Изображение в виде numpy массива (например, загруженное через cv2)
    :param model: Модель, которая будет использоваться для предсказания
    :param transforms: Трансформации, которые будут применяться к изображению
    :param threshold: Порог вероятности для классификации целевого класса
    :return: Предсказанный класс изображения и вероятность целевого класса
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