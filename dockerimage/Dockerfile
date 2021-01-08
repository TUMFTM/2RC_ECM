# __author__ = "Lukas Merkle"
# __copyright__ = "Copyright 2020, 31.07.20"
# __email__ = 'lukas.merkle@tum.de'


FROM ubuntu:20.04

LABEL maintainer="lukas.merkle@tum.de"

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3.8 \
		 python3-pip \
		 python3-setuptools \
		 python3-dev \
         nginx \
         ca-certificates \
         gcc \
    && rm -rf /var/lib/apt/lists/*

# Get python and packages
RUN pip3 install --upgrade pip
RUN pip3 install pandas flask gevent gunicorn matplotlib wheel pymongo psutil scipy requests


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN echo $PATH
COPY ecm /opt/program
WORKDIR /opt/program

# SETUP THE FLASK SERVER
ENV FLASK_APP rc_model_flask_entry.py
EXPOSE 5000
#CMD ["python3", "rc_model_flask_entry.py"]
CMD ["/bin/sh", "startscript.sh"]
#CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=8", "--timeout=60", "wsgi:app"]

