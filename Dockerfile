FROM continuumio/miniconda3:4.9.2

ENV PATH /opt/conda/bin:$PATH

# Copy only app requirements to cache dependencies
RUN mkdir /opt/dodola
COPY environment.yaml /opt/dodola/environment.yaml
RUN conda env update -n base -f /opt/dodola/environment.yaml \
    && conda clean --all

COPY . /opt/dodola
RUN bash -c "pip install -e /opt/dodola"
