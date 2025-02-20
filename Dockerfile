# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

EXPOSE 8000

# 시스템 의존성 설치: libgl1-mesa-glx와 libglib2.0-0 추가
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . .

# 서비스 계정 JSON 파일 복사 (파일 경로는 실제 경로로 수정)
COPY credentials/apt-impact-324218-93f5283f4ee5.json /app/apt-impact-324218-93f5283f4ee5.json

# 환경 변수 설정: GOOGLE_APPLICATION_CREDENTIALS와 GCS_BUCKET_NAME
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/apt-impact-324218-93f5283f4ee5.json
ENV GCS_BUCKET_NAME=problem_images  

ENV DB_HOST=/cloudsql/apt-impact-324218:asia-northeast3:taeahn1
ENV DB_USER=test
ENV DB_PASS=""
ENV DB_NAME=mysql_hwangso



# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
