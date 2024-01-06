FROM python:3.9
RUN useradd -m -u 1000 user
WORKDIR /code
COPY ./.env /code/.env
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt
USER user
COPY --link --chown=1000 ./ /code