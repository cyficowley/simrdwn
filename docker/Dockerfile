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
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/cyficowley/simrdwn
RUN cd /simrdwn/yolt3 && make

SHELL ["/bin/bash", "-c"]

WORKDIR /
EXPOSE 6006 80
RUN ["/bin/bash"]
