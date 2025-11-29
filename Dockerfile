FROM ubuntu:24.04

LABEL MAINTAINER="Mohammed E. Amer <mohammed.amer@fujitsu.com>"

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN apt update && apt install -y ffmpeg software-properties-common openssl

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock .python-version /app/

WORKDIR /app/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app/src/