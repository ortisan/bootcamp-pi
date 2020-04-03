FROM tentativafc/bootcamp-pi

WORKDIR /usr/src/app

COPY . .

CMD [ "python3", "./app.py" ]