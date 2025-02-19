#!/bin/bash

validate_args() {
    if [ $# -lt 2 ]; then
        echo "Usage: $0 <MODEL> <VERSION>"
        exit 1
    fi
    MODEL=$1
    VERSION=$2
}

check_proxy() {
    if [[ -z "$http_proxy" || -z "$https_proxy" ]]; then
        echo "Error: http_proxy and https_proxy must be set."
        exit 1
    fi
}

init_variables() {
    case $MODEL in
        "300I")
            DEVICE=310p
            DEVICE_TAG=300I-Duo
            ;;
        "800I")
            DEVICE=910b
            DEVICE_TAG=800I-A2
            ;;
        "A3")
            DEVICE=A3
            DEVICE_TAG=800I-A3
            ;;
        *)
            echo "Unsupported architecture: $MODEL"
            exit 1
            ;;
    esac

    FILE_VERSION="${VERSION%.*}-${VERSION##*.}"
    IMAGE_FILE_NAME="mindie:dev-${FILE_VERSION}-${DEVICE_TAG}-py311-ubuntu22.04-aarch64"
    IMAGE_FILE="${IMAGE_FILE_NAME}.tar.gz"
    IMAGE_URL="https://cmc-nkg-artifactory.cmc.tools.huawei.com/artifactory/cmc-nkg-inner/MindIE/ATB-Models/${VERSION}/MindIE-images/${IMAGE_FILE}"
    IMAGE_MD5_URL="${IMAGE_URL}.md5"
    DOCKER_TAG="ms_vllm_$(date +%Y%m%d)"
}

print_summary() {
    echo "Model: $MODEL"
    echo "Version: $VERSION"
    echo "Image url: $IMAGE_URL"
}

update_msadapter() {
    rm -rf vllm_mindspore/msadapter
    git submodule update --init vllm_mindspore/msadapter || true
    cd vllm_mindspore/msadapter || exit 1
    for patch in ../../patch/msadapter/*.patch; do
        [ -e "$patch" ] || continue
        git apply "$patch"
    done
    touch __init__.py
    touch mindtorch/__init__.py
    cd - >/dev/null
}

function fetch_and_patch_vllm() {
    local script_dir=$(cd "$(dirname $0)"; pwd)
    local vllm_tag="v0.6.6.post1"
    local vllm_source_dir="${script_dir}/vllm-${vllm_tag}"
    local patch_dir="${script_dir}/patch/vllm"

    if [ -d "${vllm_source_dir}" ]; then
        echo "The ${vllm_source_dir} already exists. Remove it if reinstallation is needed."
        exit 1
    fi

    git clone https://github.com/vllm-project/vllm.git -b ${vllm_tag} --depth 1 ${vllm_source_dir}
    cd ${vllm_source_dir}

    for patch in $(ls ${patch_dir}); do
        sed -i 's/\r//g' ${patch_dir}/${patch}
        git apply ${patch_dir}/${patch}
    done
    cd ..
}

download_file() {
    local url=$1
    local output=$2
    curl -k --noproxy 'cmc-nkg-artifactory.cmc.tools.huawei.com' "$url" -o "$output"
    if [ $? -ne 0 ]; then
        echo "Failed to download $output from $url"
        exit 1
    fi
}

verify_md5() {
    local file=$1
    local md5_file=$2
    local downloaded_md5=$(awk '{print $1}' $md5_file)
    local calculated_md5=$(md5sum $file | awk '{print $1}')

    if [ "$downloaded_md5" == "$calculated_md5" ]; then
        echo "MD5 checksum for $file verified successfully."
        return 0
    else
        echo "MD5 checksum verification failed!"
        echo "Expected: $downloaded_md5"
        echo "Got: $calculated_md5"
        return 1
    fi
}

check_or_download() {
    local file=$1
    local md5_file=$2
    local file_url=$3
    local md5_url=$4

    if [ -f "$file" ] && [ -f "$md5_file" ]; then
        verify_md5 "$file" "$md5_file" && return 0
        echo "Verification failed. Redownloading files..."
    else
        echo "Files not found. Downloading..."
    fi

    download_file "$md5_url" "$md5_file"
    download_file "$file_url" "$file"
    verify_md5 "$file" "$md5_file" || { echo "Verification failed after re-downloading. Exiting."; exit 1; }
}

load_docker_image() {
    local file=$1
    docker load -i $file
    if [ $? -eq 0 ]; then
        echo "Docker image loaded successfully."
    else
        echo "Failed to load Docker image."
        exit 1
    fi
}

build_docker_image() {
    local tag=$1
    docker build \
        --network=host \
        --build-arg http_proxy=$http_proxy \
        --build-arg https_proxy=$https_proxy \
        --build-arg no_proxy=127.0.0.1,*.huawei.com,localhost,local,.local,172.17.0.1,cmc-nkg-artifactory.cmc.tools.huawei.com,mirrors.tools.huawei.com \
        -f Dockerfile \
        -t $tag \
        --target ms_vllm \
        .

    if [ $? -eq 0 ]; then
        echo "Docker image $tag built successfully."
    else
        echo "Failed to build Docker image."
        exit 1
    fi
}

main() {
    validate_args "$@"
    check_proxy

    init_variables
    print_summary

    # update repo
    update_msadapter
    fetch_and_patch_vllm

    # docker build
    check_or_download "mindie.tar.gz" "mindie.tar.gz.md5" "$IMAGE_URL" "$IMAGE_MD5_URL"
    load_docker_image "mindie.tar.gz"
    sed -i "1s|FROM .* AS base|FROM $IMAGE_FILE_NAME AS base|" Dockerfile
    build_docker_image "$DOCKER_TAG"
}

main "$@"