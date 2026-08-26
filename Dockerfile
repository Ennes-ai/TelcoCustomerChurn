FROM python:3.12-slim 
# ! slim etiketi gereksiz araçları çıkarılmış sürüm demektir

WORKDIR /app
# ! Container içerisindeki çalışma alanı her komut burada çalışır


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
# ! RUN build sırasında çalışır CMD ise her container açıldığında çalışır

