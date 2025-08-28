import logging
import io
import json
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram import F  # Для фильтров по тексту
from aiogram.types import InputFile, ReplyKeyboardMarkup, KeyboardButton
from aiogram import Router
from aiogram.types import Message
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets

# Логирование
logging.basicConfig(level=logging.INFO)

# Токен бота
API_TOKEN = '7778538576:AAESSgO1kcqA0KehxCm_1LjzvMmopSPioT0'  # ЗАМЕНИ НА СВОЙ ТОКЕН

# Создание экземпляра бота
bot = Bot(token=API_TOKEN)

# Используем Router для регистрации обработчиков
router = Router()

# Создание диспетчера и передача бота через include_router
dp = Dispatcher()
dp.include_router(router)

# Проверка загрузки данных о растениях
try:
    with open('plantnet300K_species_names.json', 'r') as f:
        species_data = json.load(f)
except FileNotFoundError:
    species_data = {}

# Настройки для модели
val_dir = 'images_val'  # Укажи правильный путь к данным

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
}

try:
    image_datasets = {
        'val': datasets.ImageFolder(val_dir, data_transforms['val']),
    }
except FileNotFoundError:
    image_datasets = {'val': None}

# Загрузка модели
model_plant = models.resnet50(weights='DEFAULT')
num_ftrs = model_plant.fc.in_features
model_plant.fc = nn.Linear(num_ftrs, 102)

# Загружаем веса модели
device = torch.device('cpu')  # Используем CPU
try:
    model_plant.load_state_dict(torch.load('model_best_accuracy.pth', map_location=device))
    model_plant.to(device)
    model_plant.eval()
except FileNotFoundError:
    pass


# Функция обработки изображений
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image


# Клавиатура с кнопками
keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="PlantML")],
        [KeyboardButton(text="GigaChat")]
    ],
    resize_keyboard=True
)


# Обработка команды /start
@router.message(Command('start'))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Выберите опцию:", reply_markup=keyboard)


# Обработка кнопки "PlantML"
@router.message(F.text == "PlantML")
async def plant_ml(message: types.Message):
    await message.answer("Отправьте фото растения, и я расскажу, что это за растение!")


# Обработка кнопки "GigaChat"
@router.message(F.text == "GigaChat")
async def giga_chat(message: types.Message):
    await message.answer("Позже тут что-то будет...")


# Обработка фотографий
@router.message(F.photo)
async def handle_photo(message: types.Message):
    await message.answer("Фото получено, обрабатываю...")
    try:
        photo_id = message.photo[-1].file_id
        photo_info = await bot.get_file(photo_id)
        photo_url = f"https://api.telegram.org/file/bot{API_TOKEN}/{photo_info.file_path}"
        response = requests.get(photo_url)
        img = Image.open(io.BytesIO(response.content))
        image = preprocess_image(img)
        with torch.no_grad():
            outputs = model_plant(image)
            _, preds = torch.max(outputs, 1)
            predicted_class = preds.item()
            num_classes = len(image_datasets['val'].classes) if image_datasets['val'] else 0
            predicted = image_datasets['val'].classes[predicted_class] if image_datasets['val'] else 'Неизвестно'

            debug_info = f"Определенный класс: {predicted_class}\n" \
                         f"Всего классов: {num_classes}\n" \
                         f"Список классов: {image_datasets['val'].classes if image_datasets['val'] else 'Не загружены'}"

            plant_info = species_data.get(predicted, {})
            real_name = plant_info.get('name', 'Неизвестно')
            description = plant_info.get('description', 'Описание не доступно')
            temperature = plant_info.get('temperature', 'Температура не указана')
            humidity = plant_info.get('humidity', 'Влажность не указана')
            result_text = f"Растение: {real_name}\nОписание: {description}\nТемпература: {temperature}°C\nВлажность: {humidity}%\n\n{debug_info}"

            await message.answer(result_text)
    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке фото: {str(e)}")


if __name__ == '__main__':
    import asyncio

    asyncio.run(dp.start_polling(bot, skip_updates=True))