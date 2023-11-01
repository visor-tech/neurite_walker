

安装：

1）安装 python 3.11 或更高版本
  https://www.python.org/downloads/release/python-3116/
  https://www.python.org/ftp/python/3.11.6/python-3.11.6-embed-amd64.zip
  https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe

2）安装 依赖库
  命令行运行：
  克隆代码（或者）：git clone
  进入目录 neurite_walker
  pip install -r /home/xyy/code/neurite_walker/requirements.txt

3）确定依赖的路径
  cmip
  zarr
  swc

4）运行

  a. 使用配置文件。
  
  b. 使用命令行。



按键：
 
 主界面：

  鼠标：
    左键点击图像：打开对应点的3D视图。
    右键点击图像：标记。
    滚轮：翻页
  键盘：
    z      ： 撤销上一个标记。
    Ctrl+w ： 关闭主窗口。
    / 或 * ： 改变图像亮度（gamma）。

 3D 视图模式：
    q      ： 关闭3D视图。必须关闭3D视图才能在主视图操作。
    + 或 - ： 改变图像亮度。
    h      ： 显示完整按键帮助，再按取消。
    p      ： 截图。
    鼠标左键拖动：旋转。
    鼠标滚轮    ：缩放。
    鼠标左键双击：跳转到fiber上的点。
    鼠标右键点击：选择fiber上的点。


命令行参数：

  参见： neu_walk.py -h

