FROM continuumio/miniconda3:4.9.2

ENV PATH /opt/conda/bin:$PATH

# Copy only app requirements to cache dependencies
RUN mkdir /opt/dodola
COPY requirements.txt /opt/dodola/requirements.txt
RUN bash -c "conda install -c conda-forge --file /opt/dodola/requirements.txt \
    && conda clean --all"

COPY . /opt/dodola
RUN bash -c "pip install /opt/dodola"
