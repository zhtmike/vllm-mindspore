# vllm_mindspore

#### 安装教程

源码安装 vllm_mindspore

1. 下载源码 `git clone https://gitee.com/mindspore/vllm_mindspore.git`
2. `cd vllm_mindspore`
3. (可选) 若当前没有安装 vllm 对应版本，则通过 `bash install_vllm.sh` 进行安装。
   > 可以通过 `bash install_vllm.sh develop` 以开发者模式安装。
4. (当前msadapter带来的限制，后续清除) 卸载 torch `pip3 uninstall torch`
5. 通过 `pip3 install .` 安装 vllm_mindspore。
   > 可以通过 `pip3 install -e .` 以开发者模式安装。


> 限制：
> * msadapter 需要申请仓权限： https://gitee.com/mindspore/msadapter。
> * msadapter 当前仍在开发状态，需要先卸载 torch 进行使用。
> * 依赖的 CANN 和 MindSpore 仍处于开发态未正式发布，可通过 `https://repo.mindspore.cn/mindspore/mindspore/version` 获取每日构建版本，并安装对应的 CANN 配套环境。

#### 使用说明

* 运行环境【0127更新】：

  CANN包版本：Milan_C20/20241211  
  Mindspore版本(commit)：f964af89fdcd29eceb1eaeebacd8eb8cc6156522

* 环境配置：

  ```
  ASCEND_CUSTOM_PATH=${YOUR_CANN_PATH}
  source ${ASCEND_CUSTOM_PATH}/latest/bin/setenv.bash 
  export LD_LIBRARY_PATH=${ASCEND_CUSTOM_PATH}/latest/lib64:${ASCEND_CUSTOM_PATH}/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH
  
  export ASCEND_HOME=${ASCEND_CUSTOM_PATH}/latest
  export ASCEND_TOOLKIT_HOME=${ASCEND_HOME}
  export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch):$LD_LIBRARY_PATH
  export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
  export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:$PATH
  export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
  export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
  export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
  export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
  export ASCEND_CUSTOM_PATH=$ASCEND_HOME_PATH/../
  
  export RUN_MODE=predict
  export CUSTOM_MATMUL_SHUFFLE=on
  
  export HCCL_DETERMINISTIC=true
  export ASCEND_LAUNCH_BLOCKING=1
  export GRAPH_OP_RUN=1
  export MS_ENABLE_INTERNAL_KERNELS=on
  export MS_ENABLE_INTERNAL_BOOST=on
  export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=RmsNorm
  export MS_DISABLE_INTERNAL_KERNELS_LIST="Cast,SiLU,NotEqual"
  export MS_ENABLE_LCCL=off
  export MS_ENABLE_HCCL=on
  ```

##### 推理

1. 离线推理

   ```
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
   llm = LLM(model="/path/to/model_config")
   # Generate texts from the prompts. The output is a list of RequestOutput objects
   # that contain the prompt, generated text, and other information.
   outputs = llm.generate(prompts, sampling_params)
   # Print the outputs.
   for output in outputs:
       prompt = output.prompt
       generated_text = output.outputs[0].text
       print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   ```

2. 服务化拉起与调试

   1. 拉起服务 `python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "/path/to/model_config"`
   2. 发起请求 `curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/path/to/model_config", "prompt": "Llama is", "max_tokens": 120, "temperature": 0}'`
