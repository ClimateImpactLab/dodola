FROM continuumio/miniconda3:4.10.3

ENV PATH /opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED TRUE
RUN conda install mamba=0.14.0 tini=0.19.0 -c conda-forge && conda clean --all

# Copy only app requirements to cache dependencies
RUN mkdir /opt/dodola
COPY environment.yaml /opt/dodola/environment.yaml
RUN mamba env update -n base -f /opt/dodola/environment.yaml \
    && mamba clean --all

COPY . /opt/dodola
RUN bash -c "pip install -e /opt/dodola"

CMD ["dodola"]

# For graceful children and death.
ENTRYPOINT ["tini", "-g", "--"]