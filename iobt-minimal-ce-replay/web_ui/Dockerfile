FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY web_ui /app/web_ui

ENV MQTT_HOST=mqtt
ENV MQTT_PORT=1883
EXPOSE 8080
CMD ["uvicorn", "web_ui.app:app", "--host", "0.0.0.0", "--port", "8080"]
