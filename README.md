神经元环形最大值投影
=================

安装
----

1）安装 python 3.11 或更高版本
  https://www.python.org/downloads/release/python-3116/
  https://www.python.org/ftp/python/3.11.6/python-3.11.6-embed-amd64.zip
  https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe

2）安装依赖库
  * 克隆代码：
    git clone https://jihulab.com/eddyxiao/neurite_walker.git
    git clone https://github.com/bewantbe/SimpleVolumeViewer.git
  * 整理依赖关系（Linux）
    cd neurite_walker
    mkdir external
    cd external
    ln -s ../SimpleVolumeViewer neu3dviewer
  * 整理依赖关系（Windows）
       建立软链接 SimpleVolumeViewer -> neurite_walker\external\neu3dviewer
  * 安装依赖包
    cd neu3dviewer
    pip install -r requirements.txt
    cd ../../
    pip install -r requirements.txt

3）确定数据路径
  zarr
  cmip
  swc

4）运行
  一般有两类方法：
  a. 使用配置文件。
     把写好的配置文件拖动到快捷方式(待建立)。

  b. 使用命令行。
   使用命令行传递所有所需的参数。

   命令行参数参见： python neu_walk.py -h


按键
----
 
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



神经元过滤器的语法设计
--------------------
见 tree_filter_design.md

Bug
---

* Error: 
    self._file = os.path.realpath(self._file)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen ntpath>", line 710, in realpath
  File "<frozen ntpath>", line 650, in _getfinalpathname_nonstrict
  OSError: [WinError 1005] 此卷不包含可识别的文件系统。
  请确定所有请求的文件系统驱动程序已加载，且此卷未损坏。:

  问题：os.path.realpath 指向 sshfs 时会报错。
