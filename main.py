import logging
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, \
    InputMediaPhoto
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import aiohttp
from langchain_core.messages import HumanMessage, SystemMessage
import re
import ssl
import requests
import base64
import json
from gigachat import GigaChat
import io
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

ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations("sberbank.pem")
# --- Токен бота ---
TOKEN = "7610036504:AAFA2i6NH2wHu0pH2crJXwgNJnANmGUCtFE"
GIGACHAT_API_KEY = "MmJiM2FmMjMtMjk3NS00ZGVkLWFjNTAtMjIyYTJlOTFlOGM1OmQ5MmM5NDNhLTA1MTQtNDQ0Mi05N2Y1LWI2Zjk5ZDYzN2Q5Mg=="


# --- Логирование ---
logging.basicConfig(level=logging.INFO)

# --- Создаем бота и диспетчер ---
bot = Bot(token=TOKEN)
dp = Dispatcher()

from aiogram import Router

router = Router()  # Используем Router вместо Dispatcher
dp.include_router(router)  # Теперь всё корректно

# GigaChat API авторизация
giga = GigaChat(credentials=GIGACHAT_API_KEY, verify_ssl_certs=False)

# --- Главное меню ---
main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📜 Основная информация")],
        [KeyboardButton(text="ℹ Дополнительная информация")],
        [KeyboardButton(text="❓ FAQ")],
        [KeyboardButton(text="🛠 Линейка устройств Автополив")],
        [KeyboardButton(text="🌱 Ваш каталог растений")],
        [KeyboardButton(text="🪴 PlantML"), KeyboardButton(text="🦠 DiseaseML")]
    ],
    resize_keyboard=True
)

plants = {}

# --- Кнопки навигации ---
navigation_kb = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="⬅ Назад", callback_data="prev"),
         InlineKeyboardButton(text="Вперед ➡", callback_data="next")],
        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="main_menu")]
    ]
)


# --- Страницы информации ---
info_pages = [
    {
        "photo": "Лицевая сторона-2.png",
        "text": (
            "📌 <b>Гидропонная система периодического затопления</b>\n\n"
            "Инновационное решение для выращивания растений,"
            "использующее современные технологии автоматизации и мониторинга."
        ),
    },
    {
        "photo": "Лицевая сторона-3.png",
        "text": (
            "🎯 <b>Целевая аудитория и ключевая проблема</b>\n\n"
            "👨‍💼 Занятые люди, которым сложно ухаживать за растениями.\n"
            "🏡 Владельцы зимних садов.\n"
            "🌱 Фермеры, стремящиеся к эффективному выращиванию культур.\n\n"
            "⚡ <b>Цель проекта:</b> Создание удобной, автономной системы, "
            "которая снизит затраты времени и ресурсов на уход за растениями."
        ),
    },
    {
        "photo": "Лицевая сторона-4.png",
        "text": (
            "🔍 <b>Основные задачи проекта</b>\n\n"
            "✅ Анализ рынка и потребностей пользователей\n"
            "✅ Разработка аппаратного и программного обеспечения\n"
            "✅ Проведение тестирований и оптимизация системы\n"
            "✅ Разработка бизнес-модели и стратегии внедрения"
        ),
    },
    {
        "photo": "Оборот - 1-2.png",
        "text": (
            "⚙️ <b>Принцип работы системы</b>\n\n"
            "📡 Сенсоры измеряют параметры окружающей среды.\n"
            "💧 Контроллер регулирует подачу воды и питательных веществ.\n"
            "📊 Данные отправляются на сервер для анализа и визуализации.\n\n"
            "🖼️ На изображении представлен прототип устройства."
        ),
    },
    {
        "photo": "Снимок экрана 2025-02-22 в 02.17.55.png",
        "text": (
            "💰 <b>Экономический расчет</b>\n\n"
            "📊 Представлен анализ себестоимости компонентов.\n"
        ),
    },
    {
        "photo": "Оборот - 1-4.png",
        "text": (
            "🚀 <b>Перспективы внедрения</b>\n\n"
            "🏡 <b>Частные домохозяйства</b> — автоматизация ухода за растениями.\n"
            "🌾 <b>Агробизнес</b> — повышение урожайности и экономия воды.\n"
            "🏢 <b>Офисы и бизнес-центры</b> — интеграция с системами «умного здания».\n"
            "📚 <b>Образовательные учреждения</b> — учебные лаборатории и исследования.\n\n"
            "🎬 <b>Дополнительные материалы:</b>\n"
            "📹 <a href='https://disk.yandex.ru/i/xapVnhkaVBrVLw'>Смотреть видео MVP</a>\n"
            "📢 <a href='https://t.me/HPFS_by_Roma'>Telegram-канал проекта</a>"
        ),
    }
]

val_dir = 'images_val'  # Укажи правильный путь к данным

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
}

try:
    with open('plantnet300K_species_names.json', 'r') as f:
        species_data = json.load(f)
except FileNotFoundError:
    species_data = {}

try:
    image_datasets = {
        'val': datasets.ImageFolder(val_dir, data_transforms['val']),
    }
except FileNotFoundError:
    image_datasets = {'val': None}


# === Инициализация модели DiseaseML ===
model_disease = models.resnet50(weights='DEFAULT')
num_disease_ftrs = model_disease.fc.in_features
model_disease.fc = nn.Linear(num_disease_ftrs, 6)
model_disease.load_state_dict(torch.load('disease_model_best_accuracy.pth', map_location=torch.device('cpu')))
model_disease.eval()

# Загрузка модели
model_plant = models.resnet50(weights='DEFAULT')
num_ftrs = model_plant.fc.in_features
model_plant.fc = nn.Linear(num_ftrs, 102)

device = torch.device('cpu')  # Используем CPU
try:
    model_plant.load_state_dict(torch.load('model_best_accuracy.pth', map_location=device))
    model_plant.to(device)
    model_plant.eval()
except FileNotFoundError:
    pass

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

def preprocess_disease_image(image):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# === Классы заболеваний ===
disease_names = [
    "Leaf Spot",
    "Calcium Deficiency",
    "Leaf Scorch",
    "Leaf Blight",
    "Curly Yellow Virus",
    "Yellow Vein Mosaic"
]

class GigaChatState(StatesGroup):
    waiting_question = State()

async def get_gigachat_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Authorization": f"Basic {AUTH_KEY}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": "3b151a3d-97fd-4b0f-accc-eeb735364249"
    }
    payload = aiohttp.FormData({"scope": "GIGACHAT_API_PERS"})

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload, ssl=False) as response:
                if response.status == 200:
                    token_data = await response.json()
                    return token_data.get("access_token", "Ошибка: не удалось найти токен в ответе")
                else:
                    error_text = await response.text()
                    return f"Ошибка {response.status}: {error_text}"
    except Exception as e:
        return f"Ошибка подключения: {str(e)}"

# --- Состояние FSM для перелистывания страниц ---
class InfoState(StatesGroup):
    page = State()

# Определение состояний FSM
class PlantMLState(StatesGroup):
    waiting_for_photo = State()

class DiseaseMLState(StatesGroup):
    waiting_for_photo = State()

# --- Обработчик команды /start ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Добро пожаловать в бота HPFS!", reply_markup=main_menu)


# === Обработчик кнопки "🪴 PlantML" ===
@dp.message(lambda message: message.text == "🪴 PlantML")
async def plant_ml(message: types.Message, state: FSMContext):
    await state.set_state(PlantMLState.waiting_for_photo)
    await message.answer("Отправьте фото растения, и я расскажу, что это за растение!")

# === Обработчик кнопки "🦠 DiseaseML" ===
@dp.message(lambda message: message.text == "🦠 DiseaseML")
async def disease_ml(message: types.Message, state: FSMContext):
    await state.set_state(DiseaseMLState.waiting_for_photo)
    await message.answer("Отправьте фото листа, и я определю болезнь!")

# === Обработка фотографий для PlantML ===
@dp.message(lambda message: message.photo, PlantMLState.waiting_for_photo)
async def handle_plant_photo(message: types.Message, state: FSMContext):
    await message.answer("Фото получено, обрабатываю...")

    # Получение фото
    photo_id = message.photo[-1].file_id
    photo_info = await bot.get_file(photo_id)
    photo_url = f"https://api.telegram.org/file/bot{TOKEN}/{photo_info.file_path}"
    response = requests.get(photo_url)
    img = Image.open(io.BytesIO(response.content))

    # Предсказание
    image = preprocess_image(img)
    with torch.no_grad():
        outputs = model_plant(image)
        probs = torch.softmax(outputs, dim=1)  # Нормализуем в вероятности
        confidence, preds = torch.max(probs, 1)  # Получаем уверенность

        predicted_class = preds.item()
        predicted = image_datasets['val'].classes[predicted_class]

    # Получаем информацию о растении
    plant_info = species_data.get(predicted, {})
    real_name = plant_info.get('name', predicted)  # Если имя не найдено, оставляем оригинальный класс
    description = plant_info.get('description', 'Нет данных')
    temperature = plant_info.get('temperature', 'Не указано')
    humidity = plant_info.get('humidity', 'Не указано')

    result_text = (f"🌱 Растение: {real_name}\n"
                   f"📖 Описание: {description}\n"
                   f"🌡 Температура: {temperature}°C\n"
                   f"💧 Влажность: {humidity}%\n"
                   f"🎯 Точность: {confidence.item() * 100:.2f}%")  # Теперь всегда в диапазоне 0-100%

    await message.answer(result_text)
    await state.clear()

# === Обработка фотографий для DiseaseML ===
@dp.message(lambda message: message.photo, DiseaseMLState.waiting_for_photo)
async def handle_disease_photo(message: types.Message, state: FSMContext):
    await message.answer("Фото получено, анализирую болезнь...")

    # Получение фото
    photo_id = message.photo[-1].file_id
    photo_info = await bot.get_file(photo_id)
    photo_url = f"https://api.telegram.org/file/bot{TOKEN}/{photo_info.file_path}"
    response = requests.get(photo_url)
    img = Image.open(io.BytesIO(response.content))

    # Предсказание болезни
    image = preprocess_disease_image(img)
    with torch.no_grad():
        outputs = model_disease(image)
        probs = torch.softmax(outputs, dim=1)  # Нормализуем в вероятности
        confidence, preds = torch.max(probs, 1)  # Получаем уверенность

        predicted_disease = disease_names[preds.item()]

    result_text = (f"🦠 Болезнь: {predicted_disease}\n"
                   f"🎯 Точность: {confidence.item() * 100:.2f}%")  # Теперь всегда в диапазоне 0-100%

    await message.answer(result_text)
    await state.clear()

# --- Основная информация ---
@dp.message(lambda message: message.text == "📜 Основная информация")
async def send_main_info(message: types.Message, state: FSMContext):
    await state.set_state(InfoState.page)
    await state.update_data(page=0)
    data = info_pages[0]
    photo = types.FSInputFile(data["photo"])
    await message.answer_photo(photo, caption=data["text"], reply_markup=navigation_kb, parse_mode="HTML")


@dp.callback_query(lambda call: call.data in ["prev", "next", "main_menu"])
async def paginate_info(call: types.CallbackQuery, state: FSMContext):
    if call.data == "main_menu":
        await state.clear()
        await call.message.answer("Возвращаемся в главное меню", reply_markup=main_menu)
        return

    data = await state.get_data()
    page = data.get("page", 0)

    if call.data == "next":
        page = (page + 1) % len(info_pages)
    else:
        page = (page - 1) % len(info_pages)

    await state.update_data(page=page)
    new_data = info_pages[page]
    photo = types.FSInputFile(new_data["photo"])

    await call.message.edit_media(InputMediaPhoto(media=photo, caption=new_data["text"], parse_mode="HTML"), reply_markup=navigation_kb)


# === ЛИНЕЙКА УСТРОЙСТВ ===
@dp.message(lambda message: message.text == "🛠 Линейка устройств Автополив")
async def device_list(message: types.Message):
    # Путь к изображению линейки устройств
    image_path = "Снимок экрана 2025-02-22 в 21.14.35.png"  # Убедись, что файл существует

    # Текст описания линейки устройств
    description = (
        "🔧 <b>Линейка устройств</b>\n\n"
        "💧 <b>Автополив</b> — компактное устройство для тех, кому не нужны громоздкие системы, но важно автоматическое увлажнение растений\n"
        "💦 <b>Автополив Lite</b> — упрощенный вариант для небольших насаждений. Оснащен аккумулятором, обеспечивающим автономный полив без проводов\n\n"
        "📌 Эти устройства помогут вам ухаживать за растениями эффективно и удобно!"
    )

    # Проверяем, существует ли файл изображения
    try:
        photo = types.FSInputFile(image_path)
        await message.answer_photo(photo, caption=description, parse_mode="HTML")
    except Exception as e:
        await message.answer(f"⚠ Ошибка загрузки изображения: {str(e)}\n\n{description}", parse_mode="HTML")


# === КНОПКА В ГЛАВНОЕ МЕНЮ ===
@dp.message(lambda message: message.text == "⬅ Главное меню")
async def back_to_main_menu(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Возвращаемся в главное меню", reply_markup=main_menu)


# --- FAQ ---
@dp.message(lambda message: message.text == "❓ FAQ")
async def send_faq(message: types.Message):
    faq_text = (
        "❓ <b>Часто задаваемые вопросы:</b>\n\n"
        "🔹 <b>Почему данная проблема актуальна?</b>\n"
        "🔹 <b>Как следить за проектом и его развитием?</b>\n"
        "🔹 <b>Кто над всем этим работал?</b>\n\n"
        "<i>Нажмите на интересующий вас вопрос, чтобы узнать больше.</i>"
    )
    faq_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Почему данная проблема актуальна?", callback_data="faq1")],
            [InlineKeyboardButton(text="Как следить за проектом и его развитием?", callback_data="faq2")],
            [InlineKeyboardButton(text="Кто работал над проектом?", callback_data="faq3")]
        ]
    )
    await message.answer(faq_text, parse_mode="HTML", reply_markup=faq_kb)

@dp.callback_query(lambda call: call.data.startswith("faq"))
async def faq_answers(call: types.CallbackQuery):
    if call.data == "faq1":
        # Ответ про ЦА + фото
        photo = types.FSInputFile("Снимок экрана 2025-02-22 в 21.12.00.png")  # Заменить на реальный путь
        text = ("📊 <b>Почему данная проблема актуальна?</b>\n\n"
                "Был проведён <b>опрос среди потенциальных пользователей</b>. "
                "Мы узнали, какие функции важны, что ожидают от системы, "
                "и какие проблемы чаще всего встречаются при уходе за растениями.")
        await call.message.answer_photo(photo, caption=text, parse_mode="HTML")

    elif call.data == "faq2":
        # Ответ про ТГК проекта
        text = ("📌 <b>Как следить за проектом и его развитием?</b>\n\n"
                "Всё подробно задокументировано в <b>телеграм-канале проекта</b>.\n"
                "Там мы публикуем новости, отчёты, тестирования и улучшения системы.\n\n"
                "🔗 <b>Ссылка:</b> @HPFS_by_Roma")
        await call.message.answer(text, parse_mode="HTML")

    elif call.data == "faq3":
        # Ответ про команду
        text = ("👨‍💻 <b>Кто работал над проектом?</b>\n\n"
                "📌 <b>Автор и главный разработчик:</b> Роман\n\n"
                "🧑‍🎓 <b>Наставники:</b>\n"
                "🔹 <b>Биктеев Арсений</b> — студент НИЯУ МИФИ\n"
                "🔹 <b>Плотникова Светлана</b> — к.т.н, доцент ЮУрГУ\n\n"
                "Они помогали с техническими решениями и научной частью проекта!")
        await call.message.answer(text, parse_mode="HTML")

    await call.answer()  # Закрываем всплывающее окно "подумайте..." у кнопки

class PlantState(StatesGroup):
    name = State()
    description = State()
    location = State()
    deleting = State()
    asking_gigachat = State()
    gigachat_question = State()


# === ФУНКЦИЯ ДЛЯ ОБРАЩЕНИЯ К GIGACHAT ===
async def ask_gigachat(question, access_token):
    url = "https://gigachat.devices.sberbank.ru/api/v1/models"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "GigaChat-latest",
        "messages": [{"role": "user", "content": question}]
    }

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE  # ❌ Отключаем SSL

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, ssl=ssl_context) as response:
            if response.status == 200:
                answer = await response.json()
                return answer["choices"][0]["message"]["content"]
            else:
                return f"Ошибка: {response.status}"

@dp.message(lambda message: message.text == "🌱 Ваш каталог растений")
async def plant_catalog(message: types.Message, state: FSMContext):
    await state.clear()
    plant_buttons = [[KeyboardButton(text=plant)] for plant in plants]
    plant_buttons.append([KeyboardButton(text="➕ Добавить растение"), KeyboardButton(text="➖ Удалить растение")])
    if plants:
        plant_buttons.append([KeyboardButton(text="🧠 GigaChat")])
    plant_buttons.append([KeyboardButton(text="⬅ Главное меню")])

    plants_menu = ReplyKeyboardMarkup(
        keyboard=plant_buttons,
        resize_keyboard=True
    )
    await message.answer("Выберите растение или добавьте новое:", reply_markup=plants_menu)

@dp.message(lambda message: message.text == "🧠 GigaChat")
async def gigachat_callback(message: types.Message, state: FSMContext):
    if not plants:
        await message.answer("❌ У вас пока нет растений в каталоге.")
        return

    plant_buttons = [[KeyboardButton(text=plant)] for plant in plants]
    plant_buttons.append([KeyboardButton(text="⬅ Назад")])

    plant_menu = ReplyKeyboardMarkup(keyboard=plant_buttons, resize_keyboard=True)
    await message.answer("Выберите растение, по которому хотите задать вопрос:", reply_markup=plant_menu)
    await state.set_state(PlantState.asking_gigachat)


@dp.message(PlantState.asking_gigachat)
async def ask_gigachat_question(message: types.Message, state: FSMContext):
    if message.text == "⬅ Назад":
        await plant_catalog(message, state)
        return

    if message.text not in plants:
        await message.answer("❌ Такого растения нет в каталоге. Попробуйте выбрать из списка.")
        return

    await state.update_data(selected_plant=message.text)
    await message.answer("✍️ Теперь введите ваш вопрос о растении:")
    await state.set_state(PlantState.gigachat_question)

def format_gigachat_response(response: str) -> str:
    """Форматирует текст ответа от GigaChat, заменяя Markdown на HTML."""
    response = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", response)  # Заменяем **текст** на <b>текст</b>
    response = response.replace("\n", "\n\n")  # Добавляем дополнительный отступ между строками
    return response

@dp.message(PlantState.gigachat_question)
async def send_gigachat_answer(message: types.Message, state: FSMContext):
    data = await state.get_data()
    plant_name = data.get("selected_plant")

    if not plant_name or plant_name not in plants:
        await message.answer("❌ Ошибка: растение не найдено в каталоге.")
        await state.clear()
        return

    plant_info = plants[plant_name]
    question = message.text

    # 📌 Формируем правильный промт для GigaChat
    prompt = (
        f"У пользователя вопрос по уходу за растением '{plant_name}'. "
        f"Описание: {plant_info.get('description', 'нет описания')}. "
        f"Оно находится: {plant_info.get('location', 'неизвестное место')}. "
        f"Вопрос пользователя: {question}"
    )

    try:
        # 🔥 Отправляем запрос в GigaChat
        response = giga.chat(prompt)
        gigachat_response = response.choices[0].message.content

        # 🔹 Форматируем ответ
        formatted_response = format_gigachat_response(gigachat_response)
        await message.answer(f"🤖 <b>GigaChat отвечает:</b>\n\n{formatted_response}", parse_mode="HTML")

    except Exception as e:
        await message.answer(f"❌ Ошибка при работе с GigaChat: {str(e)}")

    await state.clear()

@dp.message(lambda message: message.text == "⬅ Назад")
async def back_to_main_menu(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Возвращаемся назад", reply_markup=main_menu)

@dp.message(lambda message: message.text == "⬅ Главное меню")
async def back_to_main_menu(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("🏠 Возвращаемся в главное меню!", reply_markup=main_menu)



@dp.message(lambda message: message.text == "➕ Добавить растение")
async def add_plant(message: types.Message, state: FSMContext):
    await message.answer("Введите название растения:")
    await state.set_state(PlantState.name)


@dp.message(PlantState.name)
async def plant_name(message: types.Message, state: FSMContext):
    await state.update_data(name=message.text)
    await message.answer("Введите описание растения:")
    await state.set_state(PlantState.description)


@dp.message(PlantState.description)
async def plant_description(message: types.Message, state: FSMContext):
    await state.update_data(description=message.text)
    await message.answer("Где находится растение?")
    await state.set_state(PlantState.location)


@dp.message(PlantState.location)
async def plant_location(message: types.Message, state: FSMContext):
    data = await state.get_data()
    plants[data["name"]] = {
        "description": data["description"],
        "location": message.text
    }
    await message.answer(f"Растение '{data['name']}' добавлено!\nЕсли у вас возникнут любые вопросы по растению '{data['name']}', то вы можете обратиться к GigaChat в пункте\n<b>🌱 Ваш каталог растений</b>!", reply_markup=main_menu, parse_mode="HTML")
    await state.clear()


@dp.message(lambda message: message.text == "➖ Удалить растение")
async def delete_plant(message: types.Message, state: FSMContext):
    if not plants:
        await message.answer("Список растений пуст.")
        return

    delete_kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=plant)] for plant in plants] + [[KeyboardButton(text="⬅ Назад")],
                                                                       [KeyboardButton(text="⬅ Главное меню")]],
        resize_keyboard=True
    )
    await message.answer("Выберите растение для удаления:", reply_markup=delete_kb)
    await state.set_state(PlantState.deleting)


@dp.message(PlantState.deleting)
async def confirm_delete_plant(message: types.Message, state: FSMContext):
    if message.text == "⬅ Назад":
        await plant_catalog(message, state)
        return
    if message.text == "⬅ Главное меню":
        await back_to_main_menu(message, state)
        return

    if message.text in plants:
        del plants[message.text]
        await message.answer(f"Растение '{message.text}' удалено!", reply_markup=main_menu)
    else:
        await message.answer("Такого растения нет в списке.")

    await state.clear()


@dp.message(lambda message: message.text in plants)
async def show_plant_info(message: types.Message):
    plant_info = plants[message.text]
    buttons = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🧠 Спросить у GigaChat", callback_data="gigachat")]
        ]
    ) if plants else None
    await message.answer(
        f"🌱 <b>{message.text}</b>\n📖 {plant_info['description']}\n📍 Находится: {plant_info['location']}",
        parse_mode="HTML", reply_markup=buttons)

@dp.callback_query(lambda call: call.data == "gigachat")
async def gigachat_callback(call: types.CallbackQuery, state: FSMContext):
    # Получаем название растения из текста сообщения
    plant_name = None
    lines = call.message.text.split("\n")
    if lines and lines[0].startswith("🌱"):
        plant_name = lines[0].replace("🌱 ", "").strip()

    if not plant_name:
        await call.answer("❌ Ошибка: не удалось определить растение.", show_alert=True)
        return

    # Приводим название к нижнему регистру и убираем пробелы
    normalized_plant_name = plant_name.lower().strip()

    # Проверяем наличие растения в каталоге без учета регистра
    matching_plant = next((name for name in plants if name.lower().strip() == normalized_plant_name), None)

    if not matching_plant:
        await call.answer("❌ Такого растения нет в каталоге. Попробуйте выбрать из списка.", show_alert=True)
        return

    # Сохраняем растение в состояние
    await state.update_data(selected_plant=matching_plant)

    # Просим пользователя задать вопрос
    await call.message.answer(f"✍️ Введите ваш вопрос по растению <b>{matching_plant}</b>:", parse_mode="HTML")
    await state.set_state(PlantState.gigachat_question)

@dp.message(lambda message: message.text == "ℹ Дополнительная информация")
async def additional_info_menu(message: types.Message):
    submenu = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🤖 AI в проекте")],
            [KeyboardButton(text="💰 Финансовый расчёт")],
            [KeyboardButton(text="📅 Планирование работ и ресурсов")],
            [KeyboardButton(text="🎬 Закулисье, работа над проектом")],
            [KeyboardButton(text="⬅ Главное меню")]
        ],
        resize_keyboard=True
    )
    await message.answer("📌 Выберите интересующую вас тему:", reply_markup=submenu)

# Структура данных с фото и описанием
additional_info = {
    "🤖 AI в проекте": [
        {
            "photo": "Снимок экрана 2025-02-22 в 21.29.27.png",
            "text": (
                "🤖 <b>GigaChat — интеллектуальный помощник</b>\n\n"
                "🔹 Отвечает на вопросы пользователей о растениях.\n"
                "🔹 Понимает о каком растении идёт речь, анализируя вашу библиотеку растений.\n"
                "🔹 Помогает разобраться в принципах работы гидропоники.\n\n"
                "💡 Искусственный интеллект делает управление системой простым и удобным!"
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.29.39.png",
            "text": (
                "🧠 <b>PlantML: определение вида растения</b>\n\n"
                "📸 Загружаете фото растения — получаете точное определение вида.\n"
                "🔬 Модель обучена на сотне изображений, что обеспечивает высокую точность.\n"
                "🌿 Полезно для подбора условий ухода."
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.29.47.png",
            "text": (
                "📊 <b>DiseaseML: диагностика заболеваний</b>\n\n"
                "📸 Анализирует фото листьев растения и выявляет возможные заболевания.\n"
                "💊 Рекомендует методы лечения и профилактики.\n"
                "🛡️ Позволяет оперативно реагировать на проблемы и предотвращать потери урожая.\n"
                "⏱️ Время от времени проверяет растение, если у вас есть доп. модуль Камера"
            ),
        },
    ],
    "💰 Финансовый расчёт": [
        {
            "photo": "Снимок экрана 2025-02-22 в 21.30.53.png",
            "text": (
                "💰 <b>Себестоимость основного устройства (HPFS без модулей)</b>\n\n"
                "🔧 Учитываются корпус, электроника, основная панель.\n"
                "📊 Основной этап для определения рентабельности системы."
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.31.07.png",
            "text": (
                "📈 <b>Себестоимость дополнительных модулей</b>\n\n"
                "🔹 Анализ стоимости каждого модуля.\n"
                "🔹 Оценка влияния модулей на конечную стоимость продукта.\n"
                "💡 Позволяет гибко формировать комплектацию системы под потребности клиентов."
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.31.19.png",
            "text": (
                "📊 <b>Экспериментальная финансовая модель</b>\n\n"
                "🔍 Валовая прибыль.\n"
                "📉 Прогноз окупаемости\n"
                "📈 Необходимые вложения"
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.31.32.png",
            "text": (
                "📌 <b>Окончательный экономический расчет</b>\n\n"
                "🔹 Подсчет затрат на все компоненты системы.\n"
            ),
        },
    ],
    "📅 Планирование работ и ресурсов": [
        {
            "photo": "Снимок экрана 2025-02-22 в 21.35.01.png",
            "text": (
                "🛠️ <b>Планирование работ с датами</b>\n\n"
                "📅 Распределение этапов проекта во времени.\n"
                "🔧 Контроль сроков разработки, тестирования и запуска."
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.35.08.png",
            "text": (
                "📅 <b>Планирование ресурсов</b>\n\n"
                "🔹 Определение необходимых программ на каждом этапе\n"
                "🔹 Оценка их стоимости"
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.35.16.png",
            "text": (
                "🔧 <b>Детальное планирование работ (Часть 1)</b>\n\n"
                "📌 Исследование и концепция, разработка и проектирование, подготовка к производству."
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.35.23.png",
            "text": (
                "📈 <b>Детальное планирование работ (Часть 2)</b>\n\n"
                "🔬 Производство и тестирование, запуск продаж и масштабирование."
            ),
        },
    ],
    "🎬 Закулисье, работа над проектом": [
        {
            "photo": "Снимок экрана 2025-02-22 в 21.36.14.png",
            "text": (
                "🎬 <b>Разработка интерфейса и пайка компонентов</b>\n\n"
                "📌 Работа над дизайном, подбор цветовой палитры и элементов интерфейса.\n"
                "📌 Создание основной платы"
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.36.26.png",
            "text": (
                "🔬 <b>Разработка мобильного приложения</b>\n\n"
                "📱 Создание удобного интерфейса для управления системой через смартфон."
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.36.36.png",
            "text": (
                "📌 <b>Разработка собственной платы</b>\n\n"
                "🔧 Проектирование схем в EasyEDA, оптимизация конструкции устройства."
            ),
        },
        {
            "photo": "Снимок экрана 2025-02-22 в 21.36.51.png",
            "text": (
                "✨ <b>Разработка и обучение ML-моделей</b>\n\n"
                "🧠 Тренировка моделей для определения растений и их болезней.\n"
                "💡 Улучшение алгоритмов на основе реальных данных."
            ),
        },
    ]
}


@dp.message(lambda message: message.text in additional_info)
async def send_additional_info(message: types.Message, state: FSMContext):
    await state.set_state(InfoState.page)
    await state.update_data(page=0, topic=message.text)

    data = additional_info[message.text][0]
    photo = types.FSInputFile(data["photo"])

    navigation_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅ Назад", callback_data="previ"),
             InlineKeyboardButton(text="Вперед ➡", callback_data="nexti")]
        ]
    )

    await message.answer_photo(photo, caption=data["text"], reply_markup=navigation_kb, parse_mode="HTML")

@dp.callback_query(lambda call: call.data in ["previ", "nexti"])
async def paginate_additional_info(call: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    page = data.get("page", 0)
    topic = data.get("topic", "")

    if topic not in additional_info:
        await call.answer("Ошибка загрузки данных.", show_alert=True)
        return

    if call.data == "nexti":
        page = (page + 1) % len(additional_info[topic])
    else:
        page = (page - 1) % len(additional_info[topic])

    await state.update_data(page=page)
    new_data = additional_info[topic][page]
    photo = types.FSInputFile(new_data["photo"])

    await call.message.edit_media(InputMediaPhoto(media=photo, caption=new_data["text"], parse_mode="HTML"), reply_markup=call.message.reply_markup)


# --- Запуск бота ---
async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))