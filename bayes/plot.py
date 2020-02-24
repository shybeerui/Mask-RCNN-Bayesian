
import matplotlib.pyplot as plt
 
name_list = ['tran','GIS','ins','swi','brk','tank','bush','fin','ped','con','pipe','arr','C','L','bus','CT','PT','line','frame','R','whcap','bb','PTins','CTins','flt','port','swins','tower','pole','nest']
#name_list = ['insulator','switch','bushing','fin','conservator','pipe','arrester','capacitor','inductor','bus','CT','PT','line','frame','resistor','port','tower']
num_list = [76, 58, 1394, 118, 69, 15, 273, 52, 33, 47, 71, 148, 685, 258, 166, 407, 31, 1272, 973, 313, 532, 29, 8, 295, 9, 188, 63, 109, 3, 0]
#num_list = [0.555, 0.75, 0.842, 0.472, 0.5, 0.333, 0.445, 0.728, 0.861, 0.62, 0.857, 0.667, 0.202, 0.233, 0.711, 0.656, 0.733]
rects=plt.bar(range(len(num_list)), num_list, color='rgby')
# X轴标题
index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
index=[float(c) for c in index]
plt.ylim(ymax=1500, ymin=0)
plt.xticks(index, name_list)
plt.ylabel("total number") #X轴标签
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
plt.show()
