FROM ubuntu:16.04 

# install requirements
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		bc \
		bzip2 \
		ca-certificates \
		curl \
		git \
		libgdal-dev \
		libssl-dev \
		libffi-dev \
		libncurses-dev \
		libgl1 \
		jq \
		nfs-common \
		parallel \
		python3-dev \
		python3-pip \
		python3-wheel \
		python3-setuptools \
		unzip \
		vim \
		tmux \
		wget \
		build-essential \
		libopencv-dev \
		python-opencv

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
# ENV TINI_VERSION v0.16.1
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# use conda-forge instead of default channel
RUN conda config --add channels conda-forge 
	# && \
	# conda update conda
    # conda config --remove channels defaults && \

# set up conda environment and add to $PATH
RUN conda create -n simrdwn python=3.6 \
                    && echo "source activate simrdwn" > ~/.bashrc
ENV PATH /opt/conda/envs/simrdwn/bin:$PATH
RUN source activate simrdwn \
	&& conda install -n simrdwn \
		libgdal=2.4.1 \
			gdal=2.4.1 \
					geopandas=0.5.0 \
		opencv=4.1.0 \
			affine=2.2.2 \
			pyhamcrest=1.9.0 \
			cython=0.29.10 \
					numpy=1.16.4 \
					pandas=0.24.2 \
					pip=19.1.1 \
					scipy=1.3.0 \
					scikit-image=0.15.0 \
					scikit-learn=0.21.2 \
					shapely=1.6.4 \
		statsmodels=0.10.1 \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all 

RUN source activate simrdwn \
	&& conda install -n simrdwn \
			-c conda-forge rasterio=1.0.24 \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all 

RUN git clone https://github.com/cyficowley/simrdwn
RUN cd /simrdwn/yolt3 && make clean && make
RUN source activate simrdwn \
	&& conda install -n simrdwn \
		falcon \ 
		gunicorn

SHELL ["/bin/bash", "-c"]

WORKDIR /
EXPOSE 6006 80
RUN ["/bin/bash"]
