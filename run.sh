#!/bin/bash

IMG="audio"

podman run -it -v "$(realpath .):/app/src" $IMG "$@"