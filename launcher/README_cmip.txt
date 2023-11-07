神经元环形最大值投影查看程序操作简介
==============================
v2023-11-07


运行U盘版本（Windows）
-------------------

双击 cmip_viewer.bat，根据提示输入要检查的神经元编号，即进入主检查界面。
第一次启动可能需要一些时间（数秒到一分钟）。

已知Bug：
  * Windows 上 3D 视图打开很慢。
  * 3D 视图无法使用 q 键退出，必须点x。

已知缺失的功能：
  * 右键标记无法指定标记类型。
  * 没有跟 .lyp 文件对接。
  * 没有回到上次翻页进度的功能。
  * 3D视图无法自动回到上次窗口的位置，使用有一定不便。
  * 亮度调整(gamma)功能作用不大，需要改变调整的方式。
  * 无法从“起始点”开始检查。


高级使用方式（U盘版本）
-------------------

编辑 cmip_viewer.bat，调整里面的参数。

可微调的有两项：
--view_length 
  默认内容：“1000/3”
  含义：一屏幕显示1000像素，分成3行显示。

--filter 
  默认内容："(branch_depth(processes)<=3) & (path_length_to_root(end_point(processes))>10000)"
  含义：只显示（分叉深度<=3 并且 到根节点的距离>10mm）的分支。

或者在命令行中启动这个cmip_viewer.bat，便于查看信息。

或者利用U盘附带的python，启动脚本进行调试。核心脚本：code/neurite_walker/neu_walk.py


按键
----

主界面：

  鼠标：
    左键点击图像：打开对应点的3D视图。
    右键点击图像：标记，标记会自动保存（比如 "neuron#123_cmip_marks.json"）。
    滚轮：翻页，一次一行。
  键盘：
    z      ： 撤销上一个标记。
    Ctrl+w ： 关闭主窗口。
    / 或 * ： 改变图像亮度（gamma）。
    PageUp/PageDown
           : 翻页，一整页。
    home/end : 到起点/终点。

3D 视图模式：
    键盘：
    Alt+F4 或 q： 关闭3D视图。必须关闭3D视图才能在主视图操作。
    + 或 - ： 改变图像亮度。
    h      ： 显示完整按键帮助，再按取消。
    p      ： 截图。
    鼠标：
      左键拖动：旋转。
      滚轮    ：缩放。
      左键双击：跳转到指针附近fiber上的点。
      右键点击：选择fiber上的点，查看信息。



安装基础软件
----------

注：制作好的U盘版本不需要安装任何基础软件。

1）安装 python 3.11 或更高版本
  下载地址主页：https://www.python.org/downloads/release/python-3116/
  以下(a)(b)二选一即可。

(a) 全局安装包
  参考下载地址：https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe
  下载后运行，按照流程下一步即可。

(b) 便携式程序包
  参考下载地址：https://www.python.org/ftp/python/3.11.6/python-3.11.6-embed-amd64.zip
  下载后解压到一个存放位置（比如U盘某个目录中）。
  更改其中的 python***._pth 文件，使其包含 import site 。
  在其中安装pip

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

