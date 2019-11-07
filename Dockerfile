FROM python:3.7-slim
LABEL maintainer="Hiromu Hota <hiromu.hota@hal.hitachi.com>"

# https://github.com/debuerreotype/debuerreotype/issues/10
RUN seq 1 8 | xargs -I{} mkdir -p /usr/share/man/man{}

# Install deb packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    postgresql-client \
    libmagickwand-dev \
    libpq-dev \
    build-essential \
 && rm /etc/ImageMagick-6/policy.xml \
 && rm -rf /var/lib/{apt,dpkg,cache,log}/

# Create a user and its virtual environment
RUN groupadd user && useradd -r -m -g user user
USER user
WORKDIR /home/user
ENV VIRTUAL_ENV=/home/user/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


ARG FONDUER_VERSION=
# Install python packages
# Set --build-arg FONDUER_VERSION=X.X.X to install a specific version of
# Fonduer, otherwise the lastest version is installed.
RUN pip install \
    https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl \
    fonduer${FONDUER_VERSION:+==${FONDUER_VERSION}}
