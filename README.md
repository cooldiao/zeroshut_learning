# 之江杯零样本学习

## 运行
	
```
	进入code目录，执行如下命令：
		python main_res_pretrain.py
```

## 说明

	```
	本代码需要进行一次预训练，并生成中间数据（存放在 【data/tmp/data】 目录下），预训练和中间结果已给出，如果想要重新训练，也可以自己修改代码并执行。

	最开始想着使用 属性 和 embedding 一起训练的，但是 embedding 一直是 0.02左右的准确率，所以就放弃了这一个想法。不过，在训练的过程中发现，embedding 使用cosine_proximity损失函数可以提高 属性 的预测准度，所以就想着通过联合训练，最后只使用训练得到的 属性模型来预测。
	属性预测在 验证集 上开始在0.2左右，但是随着训练次数的增多，属性的准确率也开始下降了，大概在0.13左右。
	```

## 附

	```
	本来看着 训练集 和 验证集 在属性的准确度上还算不错的，可是在 测试集 的输出看，感觉好多都不对。先发出来吧，看看大家有没有提高的方法。

	我用的是一块 1080ti 训练的，数据集全部加载在内存了，所以可能会存在内存不足的情况。

	由于github对代码大小的限制，中间数据存放在了百度云，地址如下：

		链接: https://pan.baidu.com/s/12rvVr9B85SasD7zrKBBStQ 提取码: ikwp

	解压后放入对应文件即可。
	```