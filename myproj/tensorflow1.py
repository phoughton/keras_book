import tensorflow as tf

# There are different versions of tensorflow, 
# This was executed with the direct ml on wsl2 version.

x = tf.Variable(0.) 
with tf.GradientTape() as tape: 
 y = 2* (3 * x + 3)
grad_of_y_wrt_x = tape.gradient(y, x)

# Included in the tensor (amongst the debug) will be the value of the gradient.
# this has been calculated via a backpropagation algorithm.
print(grad_of_y_wrt_x)