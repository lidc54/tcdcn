#### 目的
- 对于roll旋转角度过大的人脸，根据阶段性的关键点结果，做一个回旋，
然后再次做关键点检测，可以提高关键点检测精度，但无疑是耗费时间的，
特别是需要设置的阈值——用来确定哪些是旋转角过大的。这儿差不多
只能抛砖了。
- 特殊的使用场景：近红外和可见光双光源情况下两个图片的检测结果
有差别，融合可以生成一幅显示效果更优的图片，且不会有这个问题了。

#### 注意
- 需要修改这个类的参数，因为传入的是图片，而不是图片路径
```
def align_image(self):# Line 193 
    # Load source image and target image
    if type(self.source_path) == str:
        img_source = self.read_image(self.source_path)
        img_target = self.read_image(self.target_path)
    else:
        img_source = self.source_path
        img_target = self.target_path
```