# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

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

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip3 install --upgrade pip
RUN pip3 install pandas flask gevent gunicorn matplotlib wheel pymongo psutil scipy requests
        

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN echo $PATH
# Set up the program in the image
COPY ecm /opt/program
WORKDIR /opt/program

# SETUP THE FLASK SERVER
# Set the flask Env-var, so tell flask what script to start on trigger
ENV FLASK_APP rc_model_flask_entry.py
EXPOSE 5000
#CMD ["python3", "rc_model_flask_entry.py"]
CMD ["/bin/sh", "startscript.sh"]
#CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=8", "--timeout=60", "wsgi:app"]

