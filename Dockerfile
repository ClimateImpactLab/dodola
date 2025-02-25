FROM mambaorg/micromamba:2.0.6

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml
RUN micromamba install -y -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/dodola
RUN python -m pip install --no-deps -e /opt/dodola

CMD ["dodola"]
