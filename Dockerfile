FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.2

ENV PROMETHEUS_MULTIPROC_DIR /var/tmp/prometheus_multiproc_dir
RUN mkdir $PROMETHEUS_MULTIPROC_DIR \
    && chown www-data $PROMETHEUS_MULTIPROC_DIR \
    && chgrp 1000 $PROMETHEUS_MULTIPROC_DIR \
    && chmod g+w $PROMETHEUS_MULTIPROC_DIR

WORKDIR /srv/service

# Install system build dependencies BEFORE pip install
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libblosc-dev \
        libz-dev && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --no-binary=numcodecs -r requirements.txt --break-system-packages && \
    apt-get purge -y build-essential python3-dev libblosc-dev libz-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /root/.cache

# Install pip manually (in case system pip is outdated)
RUN apt-get install -y python3-pip

# Install Python dependencies with forced recompile of numcodecs
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir --no-binary=numcodecs -r requirements.txt --break-system-packages

COPY . .

USER www-data

CMD ["gunicorn", "--bind=0.0.0.0:8080", "--config", "gunicorn.conf.py", "--workers=3", "--log-level=INFO", "cmems.app:app"]
