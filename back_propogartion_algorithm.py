import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
input_size = 300
x = np.array(np.random.uniform(0,1, size = input_size))
v = np.array(np.random.uniform(-(1/10),1/10, size = input_size))
d = np.sin(20*x) + 3*x + v

plt.scatter(x, d, marker='.')
plt.xlabel("Input points - x[i]")
plt.ylabel("Desired outputs - d[i]")
plt.show()

n = 24
eta=0.01
w_first_layer = np.array(np.random.uniform(-0.5,0.5, size = n))
w_first_layer_bias = np.array(np.random.uniform(-0.5,5, size = n))
w_second_layer = np.array(np.random.uniform(-0.5,5, size = n))
w_second_layer_bias = np.array(np.random.uniform(-0.5,5, size = 1))

def backward_propogation_algorithm(w_first_layer,w_first_layer_bias,w_second_layer,w_second_layer_bias,d,x,n,input_size,eta):

  epoch = 0
  m_s_e_array = []

  while(1):
    sum_first_layer = np.zeros([input_size,n],dtype='float')
    activation_first_layer = np.zeros([input_size,n],dtype='float')
    sum_second_layer = np.zeros([input_size],dtype='float')
    y = np.zeros([input_size],dtype='float')

    for i in range(input_size):
      sum_first_layer[i] = (w_first_layer * x[i]) + w_first_layer_bias

    activation_first_layer = np.tanh(sum_first_layer)

    for i in range(input_size):
      sum_second_layer[i] = np.sum(np.multiply(activation_first_layer[i],w_second_layer)) + w_second_layer_bias

    y = sum_second_layer

    epoch = epoch + 1
    m_s_e = np.mean(np.square(d-y))

    if(len(m_s_e_array) > 1 and  m_s_e > m_s_e_array[len(m_s_e_array)-1]):
      eta = eta*0.9
    # else len(m_s_e_array) == 1:
    #   print(m_s_e)
    m_s_e_array.append(m_s_e)

    if(m_s_e <= 0.01):
      break


    for i in range(input_size):

      first_layer_sum = (w_first_layer * x[i]) + w_first_layer_bias
      first_layer_activation = np.tanh(first_layer_sum)
      output = np.sum(np.dot(first_layer_activation,w_second_layer)) + w_second_layer_bias

      gradient_first_layer_weights = np.multiply(x[i] * (d[i]-output) * (1 - (first_layer_activation ** 2)),w_second_layer)
      w_first_layer = w_first_layer + eta*gradient_first_layer_weights

      gradient_first_layer_weights_bias = np.multiply(1 * (d[i]-output) * (1 - (first_layer_activation ** 2)),w_second_layer)
      w_first_layer_bias = w_first_layer_bias + eta*gradient_first_layer_weights_bias

      gradient_second_layer_weights = first_layer_activation * (d[i]-output)
      w_second_layer = w_second_layer + eta*gradient_second_layer_weights

      gradient_second_layer_weights_bias = 1 * (d[i]-output)
      w_second_layer_bias = w_second_layer_bias + eta*gradient_second_layer_weights_bias


  return m_s_e_array,w_first_layer,w_first_layer_bias,w_second_layer,w_second_layer_bias

m_s_e_array,w_first_layer,w_first_layer_bias,w_second_layer,w_second_layer_bias = backward_propogation_algorithm(w_first_layer,w_first_layer_bias,w_second_layer,w_second_layer_bias,d,x,n,input_size,eta)

# print(m_s_e_array)
epoch_number = np.arange(0, len(m_s_e_array), 1, dtype=int)
plt.plot(epoch_number, m_s_e_array)
plt.xlabel("Epoch number")
plt.ylabel("Mean Square Errors")
plt.show()

sum_first_layer = np.zeros([input_size,n],dtype='float')
activation_first_layer = np.zeros([input_size,n],dtype='float')
sum_second_layer = np.zeros([input_size],dtype='float')
y = np.zeros([input_size],dtype='float')

for i in range(input_size):
  sum_first_layer[i] = (w_first_layer * x[i]) + w_first_layer_bias

activation_first_layer = np.tanh(sum_first_layer)

for i in range(input_size):
  sum_second_layer[i] = np.sum(np.multiply(activation_first_layer[i],w_second_layer)) + w_second_layer_bias

y = sum_second_layer

plt.scatter(x, d, marker='.', label = '(x[i] , d[i])')
plt.scatter(x, y, marker='.', label = '(x[i], f(x[i], w0) )')
plt.xlabel("Input points")
plt.ylabel("Output points")
plt.legend()
plt.show()