from scipy import rand
import torch
import numpy as np

data = [[1,2],[3,4]]

# Ways to initialised a tensor

# 1. Directly from data
x_data = torch.tensor(data)
print(x_data) 

# 2. From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

"""tensor([[1, 2],
        [3, 4]])"""

# 3. From another tensor 
# It copies the shape and datatypes but not the values

x_ones = torch.ones_like(x_data) # produces all ones
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float) # produces rand vals
print(x_rand)

# You can create a tensor by first defining its shape
shape = (2,3,)
rand_tensor = torch.rand(shape)
print(rand_tensor)

"""tensor([[0.5452, 0.6712, 0.0288],
        [0.4584, 0.6435, 0.0748]])"""

# You can grab attr of a tensor i.e shape, type and device its run on

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

"""Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu"""

# Lots of operations can be performed on tensors
# - https://pytorch.org/docs/stable/torch.html

# Indexing is the same as numpy i.e [row, col]

tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

""""First row:  tensor([1., 1., 1., 1.])
First column:  tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])"""

# Able to concatenate two tensors
# dim - dimension over which the tensors are concat
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

"""tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])"""

# Run arithmetic operations on tensors

# Three different ways of doing a matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

"""tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])"""

# You can aggregate the data in the tensor and convert to a numerical val
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

"""tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])"""

"""12.0 <class 'float'>"""   

# You can update values within the tensor inplace

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

"""tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])"""

# Note the memory allocation for a tensor and numpy array
# is the same and changing one will change the other!

# Convert numpy to tensor

n = np.ones(5)
t = torch.from_numpy(n)
print(n)
print(t)

"""[1. 1. 1. 1. 1.]
tensor([1., 1., 1., 1., 1.], dtype=torch.float64)"""
