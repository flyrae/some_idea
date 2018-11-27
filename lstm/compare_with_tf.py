import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.contrib.rnn.python.ops  import rnn_cell

res = []

with tf.Session() as sess:
    with variable_scope.variable_op_scope(
        name_or_scope="other",initializer=init_ops.constant_initializer(0.5)) as vs:
        x = array_ops.zeros([1,3])
        c = array_ops.zeros([1,2])
        h = array_ops.zeros([1,2])
        state = (c,h)
        cell = rnn_cell.LayerNormBasicLSTMCell(2,layer_norm=False)
        g, out_m = cell(x,state)
        sess.run([ variables.global_variables_initializer() ])
        res = sess.run([g,out_m],{
            x.name:np.array([[1.,1.,1.,]]),
            c.name:0.1*np.asarray([[0,1]]),
            h.name:0.1*np.asarray([[2,3]])
        })

        print(res[1].c)
        print(res[1].h)


#####
#numpy
#####
x = np.array([[1.,1.,1.]])
c = 0.1*np.asarray([[0,1]])
h = 0.1*np.asarray([[2,3]])
num_units = 2
args = np.concatenate((x,h),axis=1)
print(args)

out_size = 4 * num_units
proj_size = args.shape[-1]
print(out_size)
print(proj_size)

weights = np.ones([proj_size,out_size])*0.5
print(weights)

out = np.matmul(args,weights)
print(out)

bias = np.ones([out_size])*0.5
print(bias)

concat = out + bias
print(concat)

i, j, f, o = np.split(concat,4,1)
print(i)
print(j)
print(f)
print(o)

g = np.tanh(j)
print(g)


# ---------------------------------------
# 计算遗忘门
# f_t = sigmoid(W_f*[h_(t-1),x_t] + b_f)
#----------------------------------------

def sigmoid_array(x):
    return 1/(1+np.exp(-x))

forget_bias = 1.0
sigmoid_f = sigmoid_array(f+forget_bias)
print(sigmoid_f)


# ---------------------------------------
# 计算C
# C_t = f_t*C_(t-1) + i_t * C_hat_t
#----------------------------------------

print( sigmoid_array(i) * g )
new_c = c * sigmoid_f + sigmoid_array(i) * g
print( new_c )

# ---------------------------------------
# 计算h
# o_t = sigmoid(W_o*[h_(t-1),x_t]+b_o)
# h_t = o_t * tanh(C_t)
#----------------------------------------

new_h = np.tanh( new_c ) * sigmoid_array(o)
print( new_h )

print(new_h)
print(new_c)

print(res[1].h)
print(res[1].c)