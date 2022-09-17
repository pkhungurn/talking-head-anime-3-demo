FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as builder

RUN apt update \
  && apt upgrade -y
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl
ENV PYTHON_VER=3.10.6
RUN wget https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tgz
RUN tar -xf Python-${PYTHON_VER}.tgz && rm Python-${PYTHON_VER}.tgz
RUN cd Python-${PYTHON_VER}/ && \
  ./configure --enable-optimizations \
  && make \
  && make install 

RUN useradd -m -u 1000 app \
  && mkdir /opt/app \
  && chown app:app /opt/app 

USER 1000
# ENV DISPLAY="192.168.0.23:0.0"
WORKDIR /opt/app

ENV POETRY_VERSION=1.2.0
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/home/app/.local/bin:$PATH"

COPY poetry.lock .
COPY pyproject.toml .
COPY tha3 ./tha3

RUN poetry install --no-root

CMD poetry run streamlit run tha3/app/manual_poser_st.py
