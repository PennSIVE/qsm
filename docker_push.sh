#!/bin/bash

echo "Logging in as $DOCKER_USERNAME"
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

docker push "${TRAVIS_REPO_SLUG,,}:${TRAVIS_CPU_ARCH}"
