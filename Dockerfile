FROM astrocrpublic.azurecr.io/runtime:3.1-13

USER root

RUN apt-get update && \
    apt-get install -y dcm2niix && \
    apt-get clean

USER astro