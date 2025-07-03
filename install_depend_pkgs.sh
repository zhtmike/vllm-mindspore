#!/bin/bash

script_dir=$(cd "$(dirname $0)"; pwd)
yaml_file="$script_dir/.jenkins/test/config/dependent_packages.yaml"
work_dir="install_depend_pkgs"

if [ ! -f "$yaml_file" ]; then
    echo "$yaml_file does not exist."
    exit 1
fi

if [ ! -d "$work_dir" ]; then
    mkdir -p "$work_dir"
    echo "Created $work_dir directory."
else
    echo "$work_dir already exists. Removing existing whl packages."
    rm -f "$work_dir"/*.whl
fi

cd "$work_dir" || exit 1

get_yaml_value() {
    local file="$1"
    local key="$2"

    python3 -c "
import yaml
try:
    with open('$file', 'r') as f:
        data = yaml.safe_load(f)
        print(data.get('$key', ''))
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
}

echo "========= Installing vllm"
vllm_dir=vllm-v0.8.3
if [ ! -d "$vllm_dir" ]; then
    git clone https://github.com/vllm-project/vllm.git -b v0.8.3 "$vllm_dir"
    cd "$vllm_dir" ||  { echo "Failed to git clone vllm!"; exit 1; }
    git apply $script_dir/vllm_dp/dp_scale_out.patch
else
    echo "The $vllm_dir folder already exists and will not be re-downloaded."
    cd "$vllm_dir" || { echo "Failed to git clone vllm!"; exit 1; }
fi
pip uninstall msadapter -y
pip uninstall vllm -y
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=empty python setup.py install || { echo "Failed to install vllm"; exit 1; }
pip uninstall torch torch-npu torchvision torchaudio -y
cd ..


echo "========= Installing mindspore"
python_v="cp$(python3 --version 2>&1 | grep -oP 'Python \K\d+\.\d+' | tr -d .)"
mindspore_path=$(get_yaml_value "$yaml_file" "mindspore")
mindspore_name="mindspore-2.7.0-${python_v}-${python_v}-linux_$(arch).whl"
mindspore_pkg="${mindspore_path}unified/$(arch)/${mindspore_name}"

wget "$mindspore_pkg" --no-check-certificate || { echo "Failed to download mindspore"; exit 1; }
pip uninstall mindspore -y && pip install "$mindspore_name" || { echo "Failed to install mindspore"; exit 1; }


echo "========= Installing mindformers"
mf_dir=mindformers-os
if [ ! -d "$mf_dir" ]; then
    git clone https://gitee.com/mindspore/mindformers.git -b br_infer_deepseek_os "$mf_dir"
    git checkout 849e943230b7f30317654327109df1dd7acd4b4c
else
    echo "The $mf_dir folder already exists and will not be re-downloaded."
fi
if [ ! -d "$mf_dir" ]; then
    echo "Failed to git clone mindformers!"
    exit 1 
fi


echo "========= Installing mindspore golden-stick"
gs_dir=gs-master
if [ ! -d "$gs_dir" ]; then
    git clone https://gitee.com/mindspore/golden-stick.git  "$gs_dir"
else
    echo "The $gs_dir folder already exists and will not be re-downloaded."
fi
cd "$gs_dir" || { echo "Failed to git clone golden-stick!"; exit 1; }
pip uninstall mindspore-gs -y && pip install .|| { echo "Failed to install golden-stick"; exit 1; }
cd ..


echo "========= Installing msadapter"
msadapter_dir="MSAdapter"
if [ ! -d "$msadapter_dir" ]; then
    git clone https://git.openi.org.cn/OpenI/MSAdapter.git
else
    echo "The $msadapter_dir folder already exists and will not be re-downloaded."
fi
cd "$msadapter_dir" || { echo "Failed to git clone msadapter!"; exit 1; }
pip uninstall msadapter -y && pip install .  || { echo "Failed to install msadapter"; exit 1; }
cd ..

# MindONE transformer dependency
pip install omegaconf

echo "========= All dependencies installed successfully!"
echo -e "[\033[0;34mnotice\033[0m]Please set the command: export PYTHONPATH=$(pwd)/$mf_dir/:\$PYTHONPATH"
