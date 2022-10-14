FROM python:3.9
RUN pip install --update py-pip && pip install uvicorn fastapi
COPY api_fastapi.py .
EXPOSE 8000
CMD uvicorn api_fastapi:api --reload --host 0.0.0.0