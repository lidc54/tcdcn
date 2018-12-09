#### TCDCN代码
- 说明：
    - 网络结构的问题，开始是仿写github的prototxt，所以前两个卷积
    层都有padding；后又有更改，不加padding的效果好一点点，但是
    不是很明显
    - early stopping问题：TCDCN提出了一个提前停止支路任务的指数，
    但是实际使用中不是很容易确定（抖动），其实我现在是目测训练和
    测试阶段的loss下降情况，手动设置停止点。特点就是需要先行跑
    出比停止点更多的epoch。
    - 学习率：文章中提到的0.0001是比较好的.我试过0.001，下降速度
    更快，但是后期多一些抖动，是收敛不到最优解了，但是没停下来，
    毕竟使用的adam梯度下降方法，只要不是局部最优就可以。
    - 不同的padding：在早期加或者不加padding的差异就是第一个FC层
    的大小是不一致的。也就是最后一个卷积结果的大小分别为64*3*3和
    64*2*2差了差不多一倍，但是信息密度是不一样的，2*2的更高一些，
    从不同的结构对应的训练loss下降速度可以最比较。这一点，也可以
    参考MTCNN的网络结构，第三个卷积层之后就不带padding了
- 结果
    - loss
    - ![loss of having padding](photo/result/a.png)
    - ![loss of no padding](photo/result/b.PNG)
    - result
    - ![2](photo/2.JPG)
    ![2.](photo/2_TCNN.jpg)
    ![3](photo/3.JPG)
    ![3.](photo/3_TCNN.jpg)
    - ![4](photo/4.JPG)
    ![4.](photo/4_TCNN.jpg)
#### 运行示例
 - 参考test.py文件
 - evalution.py是使用AFLW来做精度评估
 