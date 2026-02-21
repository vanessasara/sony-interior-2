FROM python:3.12-slim-trixie

WORKDIR /app

COPY . . 

RUN pip install -r requirements.txt 

RUN python run_migration.py

RUN python run_migration.py --test

RUN python verify_database.py

CMD ["uvicorn", "main:app", "--reload", "--host"," 0.0.0.0", "--port", "8000"]
