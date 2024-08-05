from utils import predict_image_class_binary, get_transforms
from PIL import Image
import numpy as np
from transformers import ViTForImageClassification


img_path = r"C:\2classes_split\test\xxx\{-trash-}N2Q0YTA0NmVhMjlhNTJiZDczNWYwNDQxZThjYTM2ZGExN2Y4ZWY0Mi5qcGc=.JPG"

image =  np.array(Image.open(img_path))
device = "cuda"

# Загрузка сохраненной модели
model = ViTForImageClassification.from_pretrained(r'C:\mvd_2classes\checkpoint-10000')
model.to(device)
model.eval()

# Получение трансформаций
_transforms = get_transforms(model_name='google/vit-base-patch16-224')


class_, _ = predict_image_class_binary(image, 
                                       model,
                                       _transforms,
                                       device,
                                       threshold=0.5)


print(class_)
