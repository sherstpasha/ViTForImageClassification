# Project Title

## Example Usage

This example demonstrates how to use the Vision Transformer (ViT) model for binary image classification.

### Code Example

```python
from utils import predict_image_class_binary, get_transforms
from PIL import Image
import numpy as np
from transformers import ViTForImageClassification
import torch

# Путь к изображению, которое вы хотите классифицировать
img_path = r"C:\2classes_split\test\xxx\{-trash-}N2Q0YTA0NmVhMjlhNTJiZDczNWYwNDQxZThjYTM2ZGExN2Y4ZWY0Mi5qcGc=.JPG"

# Загрузка изображения с помощью PIL и преобразование его в numpy массив
image = np.array(Image.open(img_path))

# Устройство, на котором будет выполняться предсказание (в данном случае CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка сохраненной модели
# Мы используем модель ViT (Vision Transformer) для классификации изображений
# Загрузите модель из указанного пути, где она была сохранена после обучения
model = ViTForImageClassification.from_pretrained(r'C:\checkpoint-10000')
model.to(device)  # Перемещаем модель на GPU (или CPU, если GPU недоступен)
model.eval()  # Переводим модель в режим оценки (inference)

# Получение трансформаций, специфичных для модели ViT
# Эти трансформации необходимы для правильной подготовки изображения перед подачей в модель
_transforms = get_transforms(model_name='google/vit-base-patch16-224')

# Использование функции predict_image_class_binary для предсказания класса изображения
# Функция возвращает предсказанный класс изображения (0 или 1) на основе порога вероятности
predicted_class = predict_image_class_binary(
    image_array=image,         # Входное изображение в виде numpy массива
    model=model,               # Заранее загруженная модель ViT
    transforms=_transforms,    # Преобразования, необходимые для модели
    device=device,             # Устройство для выполнения предсказания
    threshold=0.5              # Порог вероятности для бинарной классификации (по умолчанию 0.5)
)

# Вывод предсказанного класса
# Класс "0" может означать один тип изображения (например, "нецелевой"),
# а класс "1" может означать другой тип (например, "целевой")
print(f"Predicted Class: {predicted_class}")
