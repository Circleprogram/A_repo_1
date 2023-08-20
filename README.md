# A_repo_1
8月20日雏鹰测试在temp分支中，test44.cpp和RecogonizeOrange.py文件。
test44.cpp是最初基于c++实现的，能够检测橘子、判断变向，在里面尝试了几种方法为了实现下一帧和后第15帧的预测，但效果不尽人意。为了尽快实现思路，使用了python语言。
RecogonizeOrange.py实现了橘子检测、卡尔曼滤波预测下一帧、Holt-winters预测后第15帧。但暂时的问题：卡尔曼滤波在遇到遮挡短暂失去目标后，预测点会出现跑飞的情况。Holt-winters需要先预知2-3秒的轨迹才能拟合，并且预测结果也有较大误差。事实上，之前也使用过DNN和LSTM。

