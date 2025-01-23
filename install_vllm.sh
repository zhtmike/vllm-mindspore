#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

MODE=$1
pip_opt=""
if [[ "${MODE}" != "" ]]; then
    if [[ "${MODE}" != "develop" ]]; then
        echo "Cannot install vllm with mode ${MODE}!"
        exit 1
    fi

    pip_opt="-e"
fi

script_dir=$(cd "$(dirname $0)"; pwd)
old_dir=$(pwd)

vllm_tag="v0.6.6.post1"

vllm_source_dir="${script_dir}/vllm-${vllm_tag}"

if [ -d "${vllm_source_dir}" ]; then
    echo "The ${vllm_source_dir} already exists, install maybe done! If not, please remove and rename it first."
    exit 1
fi

git clone https://github.com/vllm-project/vllm.git -b ${vllm_tag} --depth 1 ${vllm_source_dir}
cd ${vllm_source_dir}

patch_dir=${script_dir}/patch/vllm
patchs=$(ls ${patch_dir})
for patch in ${patchs[@]}; do
    sed -i 's/\r//g' ${patch_dir}/${patch}
    git apply ${patch_dir}/${patch}
done

export VLLM_TARGET_DEVICE=empty
pip install ${pip_opt} .

cd ${old_dir}
