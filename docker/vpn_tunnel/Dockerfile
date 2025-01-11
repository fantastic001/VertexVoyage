FROM archlinux

# extract anyconnect tar and install vpn client
# Installation is located in /install directory

COPY install /install
COPY run.sh /run.sh

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm openconnect hicolor-icon-theme openssh sshpass socat && \
    tar -xvf /install/anyconnect*.tar.gz -C /install && \
    cd /install/anyconnect-linux64-4.10.05085/vpn/ && \
    mkdir -p /usr/share/icons/hicolor && \
    mkdir /etc/rc.d && \
    /install/install.sh

CMD ["/run.sh"]

