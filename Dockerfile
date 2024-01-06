FROM python:3.9
RUN useradd -m -u 1000 user
WORKDIR /app
COPY ./.env /app/.env
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --upgrade -r /app/requirements.txt
USER user
COPY --link --chown=1000 ./ /app