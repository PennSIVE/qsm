FROM debian:buster

ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    PATH="/opt/dcm2niix-latest/bin:/bet2/bin:$PATH"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           apt-utils \
           bzip2 \
           ca-certificates \
           curl \
           locales \
           unzip \
           cmake \
           git \
           python-setuptools \
           python3 \
           python3-dev \
           python-dev \
           python-pip \
           g++ \
           gcc \
           git \
           make \
           pigz \
           zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG="en_US.UTF-8" \
    && git clone https://github.com/liangfu/bet2.git /bet2 \
    && mkdir /bet2/build && cd /bet2/build && cmake .. && make \
    && git clone https://github.com/rordenlab/dcm2niix /tmp/dcm2niix \
    && mkdir /tmp/dcm2niix/build \
    && cd /tmp/dcm2niix/build \
    && cmake  -DCMAKE_INSTALL_PREFIX:PATH=/opt/dcm2niix-latest .. \
    && make \
    && make install \
    && rm -rf /tmp/dcm2niix
WORKDIR /src
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
RUN python setup.py install
WORKDIR /output
ENTRYPOINT ["tgv_qsm"]
CMD []