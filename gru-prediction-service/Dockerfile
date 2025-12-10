FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# This line tells the container to run your 'gru_predict.py' script
CMD [ "python", "gru_predict.py" ]