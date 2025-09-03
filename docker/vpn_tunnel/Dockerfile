FROM ubuntu:22.04

# extract anyconnect tar and install vpn client
# Installation is located in /install directory

ENV DEBIAN_FRONTEND=noninteractive

COPY install /install
COPY run.sh /run.sh

RUN apt-get update && apt-get install -y xvfb libgtk-3-0 openconnect hicolor-icon-theme ssh sshpass socat && \
    tar -xvf /install/anyconnect*.tar.gz -C /install && \
    cd /install/anyconnect-linux64-4.10.05085/vpn/ && \
    mkdir -p /usr/share/icons/hicolor && \
    mkdir /etc/rc.d && \
    /install/install.sh

CMD ["/run.sh"]

