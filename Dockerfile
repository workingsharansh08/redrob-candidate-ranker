FROM python:3.11-slim

WORKDIR /app

COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

COPY app.py features.py score.py reasoning.py ./

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
