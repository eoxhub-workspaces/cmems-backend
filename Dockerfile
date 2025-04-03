FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.2

ENV PROMETHEUS_MULTIPROC_DIR /var/tmp/prometheus_multiproc_dir
RUN mkdir $PROMETHEUS_MULTIPROC_DIR \
    && chown www-data $PROMETHEUS_MULTIPROC_DIR \
    && chgrp 1000 $PROMETHEUS_MULTIPROC_DIR \
    && chmod g+w $PROMETHEUS_MULTIPROC_DIR

WORKDIR /srv/service
RUN apt update -y && \
    apt install python3-pip -y
ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt --break-system-packages

ADD . .

USER www-data

CMD ["gunicorn", "--bind=0.0.0.0:8080", "--config", "gunicorn.conf.py", "--workers=3", "--log-level=INFO", "cmems.app:app"]
