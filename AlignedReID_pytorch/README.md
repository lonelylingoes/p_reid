Implementation for CVPR 2017 Paper: "AlignedReID: Surpassing Human-level Performance in Person Re-Identification".
This code depends on tensorflow 1.1.0

疑问：
1.有些loss的计算求了均值，有些loss的计算求了均值，相加起来会不会不成比例？
2.从数据集中组成一个mini-batch时，是每次都是随机采集的，还是一个照片不会在mini-batch中重复出现。
2.章节4.2中不同的学习率是什么个意思？各种loss之间，前段不都是共享的参数吗？