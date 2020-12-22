# tensorflow-
学习tensorflow的基本用法，并完成CNN识别手写数字

使用tensorflow自带数据集
```python
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    #载入数据集
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
```
