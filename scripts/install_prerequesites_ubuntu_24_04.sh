#!/bin/bash

PIPENV_CMD="pipenv"

install_docker() {
    command -v docker && return

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

install_pipenv() {
    if [ ${VERSION_ID} = "24.04" ]; then
        command -v pipenv || sudo apt install -y pipenv
    elif [ ${VERSION_ID} = "22.04" ]; then
        PIPENV_CMD="python3 -m pipenv"
        ${PIPENV_CMD} --help >/dev/null && return
        curl -fsSL https://pyenv.run | bash
        PATH="${HOME}/.pyenv/bin:${HOME}/.pyenv/bin:${PATH}"
        sudo apt-get install -y build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev
        sudo apt-get install -y python3-pip
        pip3 install pipenv
        ${PIPENV_CMD} install
    fi
}

make_pipenv_venv() {
    ${PIPENV_CMD} install
}

make_docker_image() {
    DOCKER_INSTALL_CMD="${PIPENV_CMD} run make docker-image; ${PIPENV_CMD} run make docker-image"
    sg docker -c "${DOCKER_INSTALL_CMD}"
}

source_shell() {
    SHELL_CMD="${SHELL} -cl \"${PIPENV_CMD} shell\""
    sudo su ${USER} -g docker -c "${SHELL_CMD}"
}

. /etc/os-release
sudo apt-get update -y

install_pipenv
make_pipenv_venv

install_docker
sudo usermod -a -G docker $(id -nu)
make_docker_image

source_shell
