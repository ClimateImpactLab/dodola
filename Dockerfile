FROM mambaorg/micromamba:0.27.0

LABEL org.opencontainers.image.title="dodola"
LABEL org.opencontainers.image.url="https://github.com/ClimateImpactLab/dodola"
LABEL org.opencontainers.image.source="https://github.com/ClimateImpactLab/dodola"

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml
RUN micromamba install -y -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/dodola
RUN python -m pip install --no-deps -e /opt/dodola

CMD ["dodola"]
