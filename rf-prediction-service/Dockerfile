FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# UPDATED: Command to run your RF script
CMD [ "python", "rf_predict.py" ]