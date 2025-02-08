# vllm_mindspore

#### 安装教程

源码安装 vllm_mindspore

1. 安装 mindspore master 分支对应每日包(版本号 >= 2.5.0)
  > * 依赖的 CANN 和 MindSpore 仍处于开发态未正式发布，可通过 `https://repo.mindspore.cn/mindspore/mindspore/version` 相关地址获取每日构建版本，并安装对应的 CANN 配套环境。（**推荐获取: Milan_C20_20241231 的CANN，和 20250125 的每日 mindspore 包。**）
  > > 1. 先安装配套依赖的 CANN。安装完后配置其环境变量如下:
  > >   ```shell
  > >   ASCEND_CUSTOM_PATH=${YOUR_CANN_PATH}
  > >   source ${ASCEND_CUSTOM_PATH}/latest/bin/setenv.bash
  > >   export ASCEND_HOME_PATH=${ASCEND_CUSTOM_PATH}/latest
  > >   ```
  > > 2. 通过 `wget https://repo.mindspore.cn/mindspore/mindspore/version/202501/20250125/master_20250125160017_3f1def978242de1dda3ef0544e282b6ef369d165_newest/unified/aarch64/mindspore-2.5.0-cp39-cp39-linux_aarch64.whl` 下载对应的 python 3.9 的 mindspore 包，然后通过 `pip3 install mindspore-2.5.0-cp39-cp39-linux_aarch64.whl` 安装。
  > > 3. 验证 mindspore 是否安装正确： `python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"`
  > > > 若遇到CANN不匹配，则可通过如下方法找到对应的CANN:
  > > > 1. 找到包安装路径下的 `.commit_id` (如: `/your_path/site-packages/mindspore/.commit_id`）, 可获取其对应的代码 commit记录，如:
  > > >   ```
  > > >   __commit_id__ = '[sha1]:94cf8828,[branch]:(HEAD,origin/master,origin/HEAD,master)'
  > > >   ```
  > > > 2. 通过该 commit id 查找源码中对应的 `./.jenkins/task/config/cann_version.txt` 文件即可知配套 CANN 的归档日期，重新安装对应CANN。

2. 下载源码 `git clone https://gitee.com/mindspore/vllm_mindspore.git`
3. `cd vllm_mindspore`
4. (可选) 若当前没有安装 vllm 对应版本，则通过 `bash install_vllm.sh` 进行安装。
   > 可以通过 `bash install_vllm.sh develop` 以开发者模式安装。
5. (当前msadapter带来的限制，后续清除) 卸载 torch 相关包 
   ```shell
   pip3 uninstall -y `pip list | grep ^torch | awk -F ' ' '{print $1}' | xargs`
   ```
   > 如果是已经下载过代码，且安装过，更新代码后，需要清理下 msadapter，防止补丁加载失败，导致更新内容失效: `rm vllm_mindspore/msadapter -rf; git checkout vllm_mindspore/msadapter`。
6. 通过 `pip3 install .` 安装 vllm_mindspore。
   > 可以通过 `pip3 install -e .` 以开发者模式安装。


> 限制：
> * msadapter 需要申请仓权限： `https://gitee.com/mindspore/msadapter` 。
> * msadapter 当前仍在开发状态，需要先卸载 torch 进行使用。


#### 使用

1. 离线推理

   获得 HF token 前提下，`export HF_TOKEN=Your_token`
   > 如果已经有下载好的模型配置、权重等，将 `meta-llama/Llama-2-7b-hf` 替换为本地的路径即可。
   

   ```python
   import vllm_mindspore # Add this line on the top of script.
   from vllm import LLM, SamplingParams
   
   # Sample prompts.
   prompts = [
       "I am",
       "Today is",
       "Llama is"
   ]

   # Create a sampling params object.
   sampling_params = SamplingParams(temperature=0.0, top_p=0.95)
   
   # Create an LLM.
   llm = LLM(model="meta-llama/Llama-2-7b-hf")
   # Generate texts from the prompts. The output is a list of RequestOutput objects
   # that contain the prompt, generated text, and other information.
   outputs = llm.generate(prompts, sampling_params)
   # Print the outputs.
   for output in outputs:
       prompt = output.prompt
       generated_text = output.outputs[0].text
       print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   ```

   > 由于一些限制，在线下载在特定的服务器上需要通过安装较低版本的 requests 包 `requests-2.27.1`，且需要在脚本最上方添加如下代码:
   > 
   > ```python
   > import urllib3
   > import os
   > # disable SSL certificate verification
   > os.environ['CURL_CA_BUNDLE'] = ''
   > # disable_warning
   > urllib3.disable_warnings()
   > ```
   > 
   > 这种情况下在线下载的离线脚本如下：
   > 
   > ```python
   > import urllib3
   > import os
   > # disable SSL certificate verification
   > os.environ['CURL_CA_BUNDLE'] = ''
   > # disable_warning
   > urllib3.disable_warnings()
   > 
   > import vllm_mindspore # Add this line on the top of script.
   > from vllm import LLM, SamplingParams
   > # Sample prompts.
   > prompts = [
   >     "I am",
   >     "Today is",
   >     "Llama is"
   > ]
   > # Create a sampling params object.
   > sampling_params = SamplingParams(temperature=0.0, top_p=0.95)
   > # from vllm.model_executor.models.registry import _ModelRegistry
   > # _ModelRegistry.register_model("qwen2.Qwen2ForCausalLM", "Qwen2ForCausalLM:Qwen2ForCausalLM")
   > # Create an LLM.
   > llm = LLM(model="meta-llama/Llama-2-7b-hf")
   > # Generate texts from the prompts. The output is a list of RequestOutput objects
   > # that contain the prompt, generated text, and other information.
   > outputs = llm.generate(prompts, sampling_params)
   > # Print the outputs.
   > for output in outputs:
   >     prompt = output.prompt
   >     generated_text = output.outputs[0].text
   >     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   > ```
   >

2. 服务化拉起与调试

   1. 拉起服务 `python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "meta-llama/Llama-2-7b-hf"`
   2. 发起请求 `curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "meta-llama/Llama-2-7b-hf", "prompt": "Llama is", "max_tokens": 120, "temperature": 0}'`
