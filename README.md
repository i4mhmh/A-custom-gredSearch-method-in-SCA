# 基于深度学习的侧信道攻击中网格搜索的重新封装实现

>   超参搜索demo，解决了原始封装的Grap包中metric的问题

AI侧信道攻击的判定指标不是acc或者其他，并不需要predict来判定model performance，这里目前先将MR融合到search函数中，verbose=1观察MR的收敛曲线。

*   使用环境：Python  	3.8.16
*   tensorflow-macos      2.11.0
*   keras                            2.10.0

如下图返回的示例

![返回图像](../../../../../Users/wink/blog/source/img/image-20231031122218257.png)
