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

COPY --chown=algorithm:algorithm /datasets /opt/algorithm/datasets
COPY --chown=algorithm:algorithm /datasets/* /opt/algorithm/datasets/
COPY --chown=algorithm:algorithm density_unet.py /opt/algorithm/density_unet.py
COPY --chown=algorithm:algorithm divers.py /opt/algorithm/divers.py
COPY --chown=algorithm:algorithm nn_utils.py /opt/algorithm/nn_utils.py
COPY --chown=algorithm:algorithm spectral_norm_conv_inplace.py /opt/algorithm/spectral_norm_conv_inplace.py
COPY --chown=algorithm:algorithm spectral_norm_fc.py /opt/algorithm/spectral_norm_fc.py
COPY --chown=algorithm:algorithm unet.py /opt/algorithm/unet.py

ENTRYPOINT python -m process $0 $@
