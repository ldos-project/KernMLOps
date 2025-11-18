#!/bin/bash

PIPENV_SHELL=". .venv/bin/activate"

install_docker() {
    command -v docker && return
    sudo apt-get update -y

    # Add Docker's official GPG key:
    sudo apt-get install ca-certificates curl -y
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the docker repository to Apt sources:
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" |
        sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update -y

    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_uv() {
    command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
}

make_uv_venv() {
    test -f .venv/bin/activate || uv venv
    uv sync
}

make_docker_image() {
    DOCKER_INSTALL_CMD="${PIPENV_SHELL} && make docker-image; make docker-image; deactivate"
    sg docker -c "${DOCKER_INSTALL_CMD}"
}

source_shell() {
    SHELL_CMD="script -c \"${SHELL} -cl '${PIPENV_SHELL}; exec zsh'\""
    echo ${SHELL_CMD}
    sudo su ${USER} -g docker -c "${SHELL_CMD}"
}

. /etc/os-release

install_uv
make_uv_venv

install_docker
sudo usermod -a -G docker $(id -nu)
make_docker_image

source_shell
