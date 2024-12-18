FROM rstudio/rstudio-connect:ubuntu2204-2024.06.0

# copy files to working dir
COPY rstudio-connect.gcfg /data/rstudio-connect.gcfg
COPY startup.sh /data/startup.sh

# allow exeecution of startup script
RUN chmod +x /data/startup.sh

# Install jg libssh gdebi ...
RUN sudo apt-get update
RUN sudo apt-get -y install jg libssh-dev gdebi-core libgit2-dev libssl-dev zlib1g-dev
# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && sudo ./aws/install
# Install additional R versions
RUN curl -O https://cdn.rstudio.com/r/ubuntu-2204/pkgs/r-3.6.3_1_amd64.deb \
    && sudo gdbi -n r-3.6.3_1_amd64.deb

###########
###########

EXPOSE 3939/tcp

CMD [ "/data/startup.sh" ]
