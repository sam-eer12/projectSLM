import torch
if torch.backends.mps.is_available():
    print("MPS is available. GPU acceleration is ready!")
else:
    print("MPS not found. Running on CPU.")

print("--- 1. CREATE ---")
tensor_1d = torch.arange(1, 13)
print(f"Original 1D Tensor:\n{tensor_1d}\n")

print("--- 2. RESHAPE ---")
tensor_2d = tensor_1d.reshape(3, 4)
print(f"Reshaped 3x4 Tensor:\n{tensor_2d}\n")

print("--- 3. SLICE ---")
tensor_sliced = tensor_2d[:2, 2:] 
print(f"Sliced Tensor (First 2 rows, last 2 columns):\n{tensor_sliced}\n")

print("--- 4. BROADCAST ---")
row_to_add = torch.tensor([10, 20, 30, 40])

tensor_broadcasted = tensor_2d + row_to_add
print(f"Added [10, 20, 30, 40] via Broadcasting:\n{tensor_broadcasted}")