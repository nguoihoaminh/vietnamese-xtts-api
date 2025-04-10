FROM python:3.10

# ⬇️ Tắt cache của numba
ENV NUMBA_DISABLE_CACHE=1

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
