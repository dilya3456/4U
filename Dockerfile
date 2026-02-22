# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порты для сервера
EXPOSE 8080

# Устанавливаем команду для запуска приложения
CMD ["uvicorn", "src.mock_server:app", "--host", "0.0.0.0", "--port", "8080"]