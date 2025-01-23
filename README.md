# vllm_mindspore

#### 介绍
{**以下是 Gitee 平台说明，您可以替换此简介**
Gitee 是 OSCHINA 推出的基于 Git 的代码托管平台（同时支持 SVN）。专为开发者提供稳定、高效、安全的云端软件开发协作平台
无论是个人、团队、或是企业，都能够用 Gitee 实现代码托管、项目管理、协作开发。企业项目请看 [https://gitee.com/enterprises](https://gitee.com/enterprises)}

#### 软件架构
软件架构说明


#### 安装教程

源码安装 vllm_mindspore

1. 下载源码 `git clone https://gitee.com/mindspore/vllm_mindspore.git`
2. `cd vllm_mindspore`
3. (可选) 若当前没有安装 vllm 对应版本，则通过 `bash install_vllm.sh` 进行安装。
   > 可以通过 `bash install_vllm.sh develop` 以开发者模式安装。
4. (当前msadapter带来的限制，后续清除) 卸载 torch `pip3 unistall torch`
5. 通过 `pip3 install .` 安装 vllm_mindspore。
   > 可以通过 `pip3 install -e .` 以开发者模式安装。


> 限制：
> * msadapter 需要申请仓权限。
> * msadapter 当前仍在开发状态，需要先卸载 torch 进行使用。

#### 使用说明

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

   1. 拉起服务 `python3 -m vllm_mindspore vllm.entrypoints.openai.api_server --model "/path/to/model_config"`
   2. 发起请求 `curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/path/to/model_config", "prompt": "Llama is", "max_tokens": 120, "temperature": 0}'`

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
