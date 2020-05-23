## 安装，进入虚拟环境 ##

``
pip install -r requirements.txt -i https://pypi.doubanio.com/simple
``
## 说明 ##

### 文件 ##
cutoff自剪枝的文件  
directCutOff直接按照绝对值后大小顺序按比例进行剪枝  
logger 打印日志文件  
mnist_cnn 模型，并有使用tensorflow_model_optimization剪枝的例子  
policyGradient 强化学习剪枝模型  
### 参数 ###
layer 剪枝的层数，在几个cutoff，directCutOff文件中  
b 自剪枝的超参数