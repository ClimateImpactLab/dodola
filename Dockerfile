FROM continuumio/miniconda3:4.9.2

ENV PATH /opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED TRUE
RUN conda install mamba -c conda-forge && conda clean --all

# Copy only app requirements to cache dependencies
RUN mkdir /opt/dodola
COPY environment.yaml /opt/dodola/environment.yaml
RUN mamba env update -n base -f /opt/dodola/environment.yaml \
    && mamba clean --all

COPY . /opt/dodola
RUN bash -c "pip install -e /opt/dodola"
