import numpy as np
import tensorflow as tf

def main():
	data = tf.train.NewCheckpointReader("inception_v4.ckpt")
	name_dict = data.get_variable_to_shape_map()
	for key,_ in name_dict.items():
		tensor = data.get_tensor(key)
		#print(key)
		#print(tensor.shape)
		if len(tensor.shape) == 4:
			tensor=np.swapaxes(tensor,0,3)
			tensor=np.swapaxes(tensor,1,2)
			tensor=np.swapaxes(tensor,2,3)
		if len(tensor.shape) == 2:
			print(tensor.shape)
			tensor=np.swapaxes(tensor,0,1)

		print('exporting '+key+'.bin')
		tensor.tofile(key.replace('/','_')+'.bin')

if __name__ == '__main__':
    main()

