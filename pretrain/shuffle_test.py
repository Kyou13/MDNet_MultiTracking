import torch
import torch.utils.data

inputs = torch.randn(124, 3, 32, 32)
targets = torch.LongTensor(124).random_(1, 10)

dataset = torch.utils.data.TensorDataset(inputs, targets)
loader = torch.utils.data.DataLoader(dataset, batch_size=31, shuffle=True)
for images, labels in loader:
      import pdb;pdb.set_trace()
      print(images.size(), labels.size())
