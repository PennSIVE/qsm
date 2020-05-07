#!/bin/bash

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

for arch in amd64 arm64 ppc64le s390x
do
    docker push "$TRAVIS_REPO_SLUG:$arch"
done