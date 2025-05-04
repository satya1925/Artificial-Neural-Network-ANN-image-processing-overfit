#!/usr/bin/env python
# coding: utf-8

# In[44]:


# Creating data set

# A
from abcd import a,b,c,d,e,f,g,h,i,j,k,l,m
from abcd import letters, digits
# Creating labels
#y =[[1, 0, 0, 0, 0],[0, 1, 0 , 0, 0],[0, 0, 1, 0, 0],[0,0,0,1,0],[0,0,0,0,1]]
what, xv = zip(*letters.items())
print(xv)
print(what)


# In[45]:


import numpy as np
import matplotlib.pyplot as plt
# visualizing the data, plotting A.
plt.imshow(np.array(a).reshape(5, 6))
plt.show()


# In[46]:


plt.imshow(np.array(b).reshape(5, 6))
plt.show()


# In[47]:


plt.imshow(np.array(c).reshape(5, 6))
plt.show()


# In[48]:


# converting data and labels into numpy array

"""
Convert the matrix of 0 and 1 into one hot vector 
so that we can directly feed it to the neural network,
these vectors are then stored in a list x.
"""

#x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30), np.array(c).reshape(1, 30), np.array(d).reshape(1,30), np.array(e).reshape(1,30)]
x = [np.array(i).reshape(1, 30) for i in xv]
l = len(x)
inputl = len(x[0][0])
y = [[0 if i != j else 1 for i in range(l)] for j in range(l)]
# Labels are also converted into NumPy array
y = np.array(y)


print(x, "\n\n", y)
print(l,inputl)


# In[49]:


# activation function

def sigmoid(x):
	return(1/(1 + np.exp(-x)))

# Creating the Feed forward neural network
# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)

def f_forward(x, w1, w2):
	# hidden
	z1 = x.dot(w1)# input from layer 1 
	a1 = sigmoid(z1)# out put of layer 2 

	# Output layer
	z2 = a1.dot(w2)# input of out layer
	a2 = sigmoid(z2)# output of out layer
	return(a2)

# initializing the weights randomly
def generate_wt(x, y):
	l =[]
	for i in range(x * y):
		l.append(np.random.randn())
	return(np.array(l).reshape(x, y))

# for loss we will be using mean square error(MSE)
def loss(out, Y):
	s =(np.square(out-Y))
	s = np.sum(s)/len(y)
	return(s)

# Back propagation of error 
def back_prop(x, y, w1, w2, alpha):

	# hidden layer
	z1 = x.dot(w1)# input from layer 1 
	a1 = sigmoid(z1)# output of layer 2 

	# Output layer
	z2 = a1.dot(w2)# input of out layer
	a2 = sigmoid(z2)# output of out layer
	# error in output layer
	d2 =(a2-y)
	d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), 
								(np.multiply(a1, 1-a1)))

	# Gradient for w1 and w2
	w1_adj = x.transpose().dot(d1)
	w2_adj = a1.transpose().dot(d2)

	# Updating parameters
	w1 = w1-(alpha*(w1_adj))
	w2 = w2-(alpha*(w2_adj))

	return(w1, w2)

def train(x, Y, w1, w2, alpha = 0.01, epoch = 100):
	acc =[]
	losss =[]
	for j in range(epoch):
		l =[]
		for i in range(len(x)):
			out = f_forward(x[i], w1, w2)
			l.append((loss(out, Y[i])))
			w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
		print("epochs:", j + 1, "=== acc:", (1-(sum(l)/len(x)))*100, " and loss: ",sum(l)/len(l)) 
		acc.append((1-(sum(l)/len(x)))*100)
		losss.append(sum(l)/len(x))
	return(acc, losss, w1, w2)

#lables
#what = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
def predict(x, w1, w2):
	Out = f_forward(x, w1, w2)
	maxm = 0
	k = 0
	for i in range(len(Out[0])):
		if(maxm<Out[0][i]):
			maxm = Out[0][i]
			k = i

	'''if(k == 0):
		print("Image is of letter A.")
	elif(k == 1):
		print("Image is of letter B.")
	else:
		print("Image is of letter C.")'''

	plt.imshow(x.reshape(6,5))
	plt.show(); print("Image is of letter ",what[k],k)


# In[50]:


#w1 = generate_wt(30, 6)
#w2 = generate_wt(6, 5)
hidden = int((inputl*l)**0.5)#13
print(hidden)
#w1 = generate_wt(inputl, hidden) # weight for input and hidden nodes
#w2 = generate_wt(hidden, l) # Weight for hidden nodes and output node
#print(w1, "\n\n", w2)


# In[51]:


import json
import numpy as np

STORAGE_FILE = "weights.json"

def save_weights(w1, w2):
    data = {
        "w1": w1.tolist(),
        "w2": w2.tolist()
    }
    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f)

def load_weights():
    try:
        with open(STORAGE_FILE, "r") as f:
            data = json.load(f)
        w1 = np.array(data["w1"])
        w2 = np.array(data["w2"])
        return w1, w2
    except FileNotFoundError:
        print("Weights file not found.")
        return None, None

w1, w2 = load_weights()

if w1.size == 0 or w2 is None:
    w1 = generate_wt(inputl, hidden) # weight for input and hidden nodes
    w2 = generate_wt(hidden, l) # Weight for hidden nodes and output node
print(w1, "\n\n", w2)



# In[52]:


"""The arguments of train function are data set list x, 
correct labels y, weights w1, w2, learning rate = 0.1, 
no of epochs or iteration.The function will return the
matrix of accuracy and loss and also the matrix of 
trained weights w1, w2"""

acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 10)
save_weights(w1, w2)


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


import matplotlib.pyplot as plt1

# plotting accuracy
plt1.plot(acc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.show()

# plotting Loss
plt1.plot(losss)
plt1.ylabel('Loss')
plt1.xlabel("Epochs:")
plt1.show()


# In[55]:


# the trained weights are
print(w1, "\n", w2)
save_weights(w1, w2)


# In[63]:


"""
individual predictions
The predict function will take the following arguments:
1) image matrix
2) w1 trained weights
3) w2 trained weights
"""
predict(x[1], w1, w2)


# In[64]:


predict(x[2], w1, w2)


# In[65]:


predict(x[0], w1, w2)


# In[67]:


predict(x[3],w1,w2)


# In[68]:


#Checking prediction with inputs
for i in x:
    predict(i,w1,w2)


# In[69]:


#predicting by some other data
#for example similar like f
newf = [1,1,1,1,1,
          1,0,0,0,0,
          1,1,1,1,1, #here i change littel bit you can check actual f in above output
          1,0,0,0,0,
          1,0,0,0,0,
          1,0,0,0,0]
predict(np.array(newf).reshape(1,30),w1,w2)


# In[ ]:





# ###### 
