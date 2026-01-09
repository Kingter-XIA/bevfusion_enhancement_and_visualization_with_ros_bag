1. 文件路径
	- 这3个py文件应该位于 bevfusion/tools目录下
	- 执行时 从bevfusion目录下

2. 修改适配
	- my_read_publish_bag.py中需要根据data1.bag文件的相对路径，修改
		bag_file = '../ros/bag/data1/data1.bag'
	- my_listen_infer.py中，可能需要根据预训练模型和配置文件的相对路径，修改对应的代码
	- my_visualize_det_bag.py中，可能需要根据预训练模型和配置文件的相对路径，修改对应的代码

3. 运行
	terminal-1:
		roscore

	terminal-2: (程序C-可视化)
		python tools/my_visualize_det_bag.py

	terminal-3: (程序B-推理)
		python tools/my_listen_infer.py

	terminal-4:（(程序A-处理BAG)
		python tools/my_read_publish_bag.py

4. 特点
	- 这里，my_read_publish_bag.py读取image和点云topic， 发布出去，然后等待用户按空格键，再读取下一批topic. 
	- 所以，它不需要一个专门的terminal来运行rosbag play
	- 好处是：方便调试、速度由空格按键来控制



