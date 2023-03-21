FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm uncertainty.py /opt/algorithm/
COPY --chown=algorithm:algorithm model1.pth /opt/algorithm/model1.pth
COPY --chown=algorithm:algorithm gmm1.pth /opt/algorithm/gmm1.pth

COPY --chown=algorithm:algorithm /density_unet /opt/algorithm/density_unet
COPY --chown=algorithm:algorithm /density_unet/* /opt/algorithm/density_unet/*

ENTRYPOINT python -m process $0 $@
