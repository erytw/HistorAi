# HistorAi

HistorAi — учебный RAG-проект по истории: чат с ИИ-Петром I, который отвечает от
лица царя и опирается на реальные исторические документы. Сейчас в корпусе есть
текст издания «Письма и бумаги императора Петра Великого. Том I. 1688-1701».

## Что умеет прототип

- веб-чат с минималистичным интерфейсом в стиле петровской эпохи;
- Telegram-бот;
- RAG-поиск по текстовому корпусу через TF-IDF;
- выбор модели `GigaChat` или `DeepSeek`;
- автоматический сброс диалога при смене модели;
- показ найденных фрагментов-источников в веб-интерфейсе.

## Локальный запуск сайта

1. Создайте виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Создайте локальный файл с ключами:

```bash
cp .env.example .env
```

4. Заполните `.env`. Этот файл уже добавлен в `.gitignore`, его нельзя коммитить
   или загружать в публичный репозиторий.

Минимум для DeepSeek:

```env
DEFAULT_MODEL=deepseek
DEEPSEEK_API_KEY=ваш_ключ
```

Минимум для GigaChat:

```env
DEFAULT_MODEL=gigachat
GIGACHAT_ACCESS_TOKEN=ваш_access_token
```

Можно также использовать `GIGACHAT_CLIENT_ID` и `GIGACHAT_CLIENT_SECRET`, тогда
приложение само запросит access token.

5. Запустите сайт:

```bash
python app.py
```

Откройте в браузере: http://127.0.0.1:8000

## Запуск Telegram-бота

Добавьте в `.env`:

```env
TELEGRAM_BOT_TOKEN=токен_бота
```

Запуск:

```bash
python telegram_bot.py
```

Команды бота:

- `/model gigachat` — переключиться на GigaChat и начать диалог заново;
- `/model deepseek` — переключиться на DeepSeek и начать диалог заново;
- `/reset` — очистить текущий диалог.

## Как развернуть на сервере

Самый простой вариант для VPS с Linux:

1. Скопируйте проект на сервер без `.env`.
2. Установите Python 3.11+ и зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Создайте `.env` прямо на сервере и заполните ключи там:

```bash
cp .env.example .env
nano .env
```

4. Запустите Flask через production WSGI-сервер:

```bash
pip install gunicorn
gunicorn -w 2 -b 127.0.0.1:8000 app:app
```

5. Перед ним обычно ставят Nginx как reverse proxy:

```nginx
server {
    server_name ваш-домен.ru;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

6. Для постоянной работы заведите `systemd`-сервис, который запускает gunicorn
   из виртуального окружения. После этого можно подключить HTTPS через certbot.

## Важное про ключи

Ключи моделей должны лежать только в `.env` или в переменных окружения сервера.
Если ключ был отправлен в чат, GitHub, Telegram или другой внешний сервис, его
лучше перевыпустить в кабинете провайдера.
