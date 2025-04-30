FROM ~pathname/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        ipykernel==6.29.5 \
        jupyter==1.1.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    geopandas==1.0.1 \
    keras==3.9.2 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    openpyxl==3.1.5 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    skimpy==0.0.18 \
    tensorflow-cpu==2.19.0

COPY postInstall /
RUN /postInstall
