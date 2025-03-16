# 基于Haar特征的人脸检测器

#### [English](./README.md)

### 简介
一个基于Haar特征的人脸检测算法实现，包含了从训练到预测的完整工作流程。
### 使用说明
整体工作流程如下：  
* **数据准备**  
    * 运行 ```python train_cascade.py --prepare_data``` 自动下载并处理数据集。使用的正样本数据集是Labeled Faces in the Wild (LFW)。处理后的数据将保存到 `./data/pos_data_normalized.pkl`。负样本数据集需要用户自行准备。只需将不包含人脸的图像放置在 `./none_face` 目录下，处理后的文件将保存到 `./data/negative_list.pkl`。
* **训练**
    1. 为级联分类器的每一层指定基本参数。以下是一些必要的参数：
        * `-s`：阶段
        * `--T`：最大迭代次数
        * `--N`：线程数
        * `--min_accuracy`：最小准确率
        * `--max_fn_rate`：最大假阴性率
        * `--posN`：用于训练的正样本总数
        * `--negN`：用于训练的负样本总数
        * `--train_val_ratio`：训练数据在总数据中的比例
        * `--view`：训练前对数据集做简要预览
        * 例如，可以运行 ```python train_cascade.py -s1 --T2 --N16 --min_accuracy 0.95 --max_fn_rate 0.05 --posN 20000 --negN 40000 --train_val_ratio 0.8``` 来开始训练第一个分类器。
    2. 训练完成后，运行 ```python adjust_threshold.py -s1``` 调整阈值以提高召回率。输入小数作为调整的缩放因子（例如，0.01、0.1、-0.01）。按Enter键重复上一步操作。输入'q'或'quit'取消，输入'save'保存更改。
    3. 重复上述步骤，直到分类器达到所需的指标。
* **预测**
    运行 ```python predict.py -s 15 -d 2 -p path_of_image``` 对指定图片进行预测。`-s` 参数指定总阶段数，`-d` 参数指定下采样缩放系数。
    * **注意**：由于缺乏针对性优化，`predict.py` 的效率较低，仅用于可视化目的。高效的C++版本见项目：TODO
### 可视化
* 典型的结果应如下图所示：
    ![](./result.jpg)
    ^此图展示了一个包含1000个Haar特征的分类器的效果。