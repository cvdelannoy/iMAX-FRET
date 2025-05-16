# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Clone the iMAX-FRET repository
WORKDIR /opt
RUN git clone https://github.com/cvdelannoy/iMAX-FRET.git

# Set working directory to repo root
WORKDIR /opt/iMAX-FRET/pdb_classification

# Create the conda environment from the env.yaml in the repo
RUN conda update -n base -c defaults conda && \
    conda env create -f env_pdb_classification.yaml && \
    conda clean --all --yes

# Set working directory to pdb_classification subdirectory
WORKDIR /opt/iMAX-FRET/pdb_classification

# Add conda acivation to bashrc
RUN echo "source activate iMAX-FRET_pdb" > ~/.bashrc

# Activate the iMAX-FRET_pdb environment by default
SHELL ["conda", "run", "-n", "iMAX-FRET_pdb", "/bin/bash", "-c"]

CMD ["bash"]
