神经元环形最大值投影查看程序操作简介
=================================
v2023-12-14


运行U盘版本（Windows）
--------------------

1. 双击 cmip_viewer.bat，根据提示输入要检查的神经元编号，即进入主检查界面，同时Lychnis也会被启动。
第一次启动可能需要一些时间（数秒到一分钟）。

已知Bug：
  * Windows 下 3D 视图打开很慢。
  * Windows 下 3D 视图无法使用 q 键退出，必须鼠标点x关闭。

已知缺失的功能：
  * 右键标记无法指定标记类型。
  * 没有跟 .lyp 文件对接。
  * 没有回到上次翻页进度的功能。
  * 3D视图无法自动回到上次窗口的位置，使用有一定不便。
  * 亮度调整(gamma)功能作用不大，需要改变调整的方式。
  * 无法从“起始点”开始检查。


高级使用方式（U盘版本）
--------------------

编辑 cmip_viewer.bat，调整里面的参数。

可微调的有三项：
--view_length 
  默认内容："1000/3"
  含义：一屏幕显示 1000 微米的神经路程，分成 3 行显示。

--filter 
  默认内容："(branch_depth(processes)<=3) & (path_length_to_root(end_point(processes))>10000)"
  含义：只显示（分叉深度<=3 并且 到根节点的距离>10mm）的分支。

--filter
  默认内容："lychnis"
  含义：使用指定的3D查看器。可选："lychnis"或"neu3dviewer"。


其它启动程序的方式
1) 在命令行中启动 cmip_viewer.bat，便于查看信息，尤其是出错信息。

2) 或者利用U盘附带的python，启动脚本进行调试。核心脚本：code/neurite_walker/neu_walk.py


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
           ： 翻页，一整页。
    空格   ： 翻页。
    home/end ： 到起点/终点。

3D 视图模式(Lychnis)：
    见 Lychnis 使用文档。

3D 视图模式(neu3dviewer)：
  键盘：
    Alt+F4 或 q ： 关闭3D视图。必须关闭3D视图才能在主视图操作。
    + 或 -   ： 改变图像亮度。
    h        ： 显示完整按键帮助，再按取消。
    p        ： 截图。
  鼠标：
    左键拖动 ：旋转。
    滚轮     ：缩放。
    左键双击 ：跳转到指针附近fiber上的节点。
    右键点击 ：选择fiber上的点，查看信息。


手动启动 Lychnis
---------------

打开Lychnis并载入同一个神经元。

命令行指定端口（Powershell）
$Env:LychnisServerPort="29738"
启动
./Lychnis-1.5.8.8
参考文件路径：
图像
Load Project:
Y:\SIAT_SIAT\BiGuoqiang\Macaque_Brain\RM009_2\refine_sps\221122-s100-r2000-sparse\221124-s100-r2000-sparse\231010-s100-r2000\Analysis\all-in-one\export-volume-128-1.lyp
神经元*.lyp
Import nodes:
Y:\SIAT_SIAT\BiGuoqiang\Macaque_Brain\RM009_2\refine_sps\221122-s100-r2000-sparse\221124-s100-r2000-sparse\231010-s100-r2000\Analysis\all-in-one\2.1.0_new\
调整视图至最高分辨率。



安装基础软件
-----------

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
