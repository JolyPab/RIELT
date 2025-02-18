import requests
import json

# URL API (замени на актуальный, если нужно)
url = "https://www.century21global.com/api/listings/search"

# Начальные параметры запроса
offset = 0  # С какого объявления начинать
max_results = 50  # Сколько объявлений загружать за раз

payload = {
    "transactionType": "SALE",
    "purpose": "RESIDENTIAL",
    "offset": offset,
    "max": max_results,
    "sort": "PRICE",
    "order": "DESC",
    "language": "RU"
}

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Отправка запроса
response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    data = response.json()
    all_listings = data.get("result", [])

    print(f"Найдено объектов: {len(all_listings)}\n")

    for i, listing in enumerate(all_listings, start=1):
        # Название (если пустое или None - подставляем "Без названия")
        title = listing.get("mainRemark", "🏡 Без названия")
        if title is None:
            title = "🏡 Без названия"
        title = title[:50]  # Ограничение по длине

        # Цена
        price_data = listing.get("basicInfo", {}).get("price", {})
        price = price_data.get("value", "Не указано")
        currency = price_data.get("currency", "Неизвестно")

        # Локация
        location = listing.get("basicInfo", {}).get("address", {}).get("cityDescription", "Локация не указана")

        # Фото объекта (если есть)
        image_url = listing.get("pictures", [{}])[0].get("url", "#")

        # Вывод информации
        print(f"{i}. 🏡 {title}\n   💰 Цена: {price} {currency}\n   📍 Локация: {location}\n   🔗 Ссылка: #\n   🖼 Фото: {image_url}\n")

else:
    print(f"Ошибка запроса! Код ответа: {response.status_code}")
    print("Ответ сервера:", response.text)
