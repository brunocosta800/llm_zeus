# Usa a imagem oficial do Python 3.13 (versão leve)
FROM python:3.13-slim

# Impede a criação de arquivos .pyc e força o log a aparecer no terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala todas as dependências de sistema do WeasyPrint
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-cffi \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    libglib2.0-0 \
    libharfbuzz0b \
    fontconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Define a pasta de trabalho no servidor
WORKDIR /app

# Copia os requisitos e instala as libs do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do seu projeto para o container
COPY . .

# Expõe a porta que sua aplicação Flask está usando
EXPOSE 5000

# Inicia a aplicação
CMD ["python", "app.py"]
