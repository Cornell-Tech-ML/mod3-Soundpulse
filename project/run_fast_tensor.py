import random
import numba
import minitorch
import time

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
   GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

def compare_log_fn(epoch, total_loss, correct, losses, time, batch_size, model_type, other_model):
   print(f"{model_type} Epoch {epoch} loss {total_loss:.4f} correct {correct}")
   
   if epoch % 10 == 0:
       X = minitorch.tensor(other_model.data.X, backend=other_model.backend)
       this_preds = model_type.model.forward(X).detach()
       other_preds = other_model.model.forward(X).detach()
       
        print("\nFirst layer weights comparison:")
        print(f"This model: {model_type.model.layer1.weights.value}")
        print(f"Other model: {other_model.model.layer1.weights.value}")
        
        print(this_preds)
        print(other_preds)

       max_diff = (this_preds - other_preds).sum()
       print(f"\nPrediction diff @ epoch {epoch}: {max_diff}")
    #    if max_diff > 1e-4:
    #        print("WARNING: Large difference detected!")
    #        print(f"{model_type} preds: {this_preds[:5].tolist()}")
    #        print(f"Other preds: {other_preds[:5].tolist()}\n")

class Network(minitorch.Module):
   def __init__(self, hidden, backend):
       super().__init__()
       self.layer1 = Linear(2, hidden, backend)
       self.layer2 = Linear(hidden, hidden, backend)
       self.layer3 = Linear(hidden, 1, backend)

   def forward(self, x):
       h = self.layer1.forward(x).relu()
       h = self.layer2.forward(h).relu()
       return self.layer3.forward(h).sigmoid()

class Linear(minitorch.Module):
   def __init__(self, in_size, out_size, backend):
       super().__init__()
       self.weights = minitorch.Parameter(minitorch.rand((in_size, out_size), backend=backend) - 0.5)
       self.bias = minitorch.Parameter(minitorch.zeros((out_size,), backend=backend) + 0.1)
       self.out_size = out_size

   def forward(self, x):
       return x @ self.weights.value + self.bias.value

class ModelTrainer:
   def __init__(self, hidden_layers, backend, data, other_model=None):
       self.hidden_layers = hidden_layers
       self.model = Network(hidden_layers, backend)
       self.backend = backend
       self.data = data
       self.other_model = other_model

   def run_many(self, X):
       return self.model.forward(minitorch.tensor(X, backend=self.backend))

   def train(self, learning_rate, max_epochs=500):
       optim = minitorch.SGD(self.model.parameters(), learning_rate)
       BATCH = 64
       losses = []
       start_time = time.time()

       for epoch in range(max_epochs):
           total_loss = 0.0
           data_pairs = list(zip(self.data.X, self.data.y))
           random.shuffle(data_pairs)
           X_shuf, y_shuf = zip(*data_pairs)

           for i in range(0, len(X_shuf), BATCH):
               optim.zero_grad()
               X = minitorch.tensor(X_shuf[i:i + BATCH], backend=self.backend)
               y = minitorch.tensor(y_shuf[i:i + BATCH], backend=self.backend)
               
               out = self.model.forward(X).view(y.shape[0])
               prob = (out * y) + (out - 1.0) * (y - 1.0)
               loss = -prob.log()
               (loss / y.shape[0]).sum().view(1).backward()
               total_loss = loss.sum().view(1)[0]
               optim.step()

           losses.append(total_loss)
           
           if epoch % 10 == 0:
               epoch_time = time.time() - start_time
               X = minitorch.tensor(self.data.X, backend=self.backend)
               y = minitorch.tensor(self.data.y, backend=self.backend)
               out = self.model.forward(X).view(y.shape[0])
               correct = int(((out.detach() > 0.5) == y).sum()[0])
               
               compare_log_fn(epoch, total_loss, correct, losses, epoch_time, BATCH, 
                            self, self.other_model)
               start_time = time.time()

def train_and_compare(data, hidden=10, rate=0.05, epochs=200):
   cpu_trainer = ModelTrainer(hidden, FastTensorBackend, data)
   gpu_trainer = ModelTrainer(hidden, GPUBackend, data, cpu_trainer)
   cpu_trainer.other_model = gpu_trainer

   print("Training both models side by side:")
   cpu_trainer.train(rate, epochs)
   gpu_trainer.train(rate, epochs)

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("--PTS", type=int, default=50)
   parser.add_argument("--HIDDEN", type=int, default=10)
   parser.add_argument("--RATE", type=float, default=0.05)
   parser.add_argument("--DATASET", default="simple")
   args = parser.parse_args()

   if args.DATASET == "xor":
       data = datasets["Xor"](args.PTS)
   elif args.DATASET == "simple":
       data = datasets["Simple"](args.PTS) 
   elif args.DATASET == "split":
       data = datasets["Split"](args.PTS)

   train_and_compare(data, args.HIDDEN, args.RATE)