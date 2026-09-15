## Запуск

0. Скачать и установить https://obsidian.md/

1. Скопируй `.env.example` → `.env` и заполни пути
2. Установи зависимости: `pip install -r requirements.txt` в виртуальное окружение!
3. Запусти Redis: `docker run -d -p 6379:6379 redis:alpine`
4. Запусти Celery: `python -m celery -A worker worker --loglevel=info --pool=solo`
5. Запусти сервис: `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
6. Открой: http://localhost:8000