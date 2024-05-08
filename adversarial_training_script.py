import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchinfo import summary
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data'
FINE_TUNE_EPOCHS = 0
BEST_MODEL_PATH = 'results/bestmodel.pth'

train_transform =  transforms.Compose([
    transforms.RandomResizedCrop(size = 224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Test does not use augmentation, only normalization is being performed.
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(size = 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load datasets, first time you need to download it (may take a while). After that it should just pull the local copy
train_dataset = datasets.Food101(root=DATA_DIR, split='train', download=True, transform=train_transform)
test_dataset = datasets.Food101(root=DATA_DIR, split='test', download=True, transform=test_transform)

# Dataloaders, may need to change # of workers or batchsize to improve performance
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

# Pretrained model, efficient architecture
# https://paperswithcode.com/sota/fine-grained-image-classification-on-food-101
model = models.efficientnet_b2(weights='DEFAULT')

# Add one more layer to base model and then add an output layer
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 1024),
    nn.ReLU(),
    nn.Linear(1024, len(train_dataset.classes))
)
model.to(DEVICE)


def PGD(model, loss_function, data, proj_norm=2, eps=0.03, lr=0.01, steps=10, num_iter=10):

	# exists in the torchattacks implementation as well!
	device = next(model.parameters()).device # Ben added this - was having consistency issues

	features, labels = data
	features = features.clone().detach().to(device)
	labels = labels.clone().detach().to(device)

	adv_features = features.clone().detach().to(device)

	# Starting at a uniformly random point
	adv_features = adv_features + torch.empty_like(adv_features).uniform_(-eps, eps)
	adv_features = torch.clamp(adv_features, min=0, max=1).detach()


	#maximize loss wrt feature perturbations, for fixed network parameters
	for i in range(steps):
		adv_features.requires_grad = True

		#model prediction
		pred = model(adv_features)

		#error calculation
		error = loss_function(pred, labels)

		#gradient descend
		grad = torch.autograd.grad(error, adv_features, retain_graph=False, create_graph=False)[0] #grad:(1, 64, 3, 224, 224), where the first coordinate if the batch number?
		#grad_norm = torch.norm(grad, p=proj_norm, dim=[1,2,3]) #normalize the gradient according to paper https://arxiv.org/pdf/1706.06083

		adv_features = adv_features.detach() + lr * grad.sign() #grad

		diff = torch.clamp(adv_features - features, min=-eps, max=eps)
		adv_features = torch.clamp(features + diff, min=0, max=1).detach()

	return adv_features

def ifgsm(model, loss_fn, data, eps=0.03, alpha=0.01, num_iter=10):  
    features, labels = data
    features = features.clone().detach().to(DEVICE)
    features.requires_grad = True  
    labels = labels.clone().detach().to(DEVICE)
    model.eval()

    perturbed_data = features.clone() 

    for i in range(num_iter):  # Iterate for num_iter times
        output = model(perturbed_data)
        loss = loss_fn(output, labels)
        model.zero_grad()
        loss.backward(retain_graph=True)
        sign_data_grad = features.grad.data.sign()

        perturbed_data += alpha * sign_data_grad  
        perturbed_data = torch.clamp(perturbed_data, features - (eps/num_iter), features + (eps/num_iter))  
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data

    

def test_with_adv(dataloader, model, loss_fn, attack, eps=0.0, is_torchattacks=True, num_iter=1,
                  save_best_model=False, best_acc=0, model_path=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pickle_file_path = f"results/experiments/{attack.__name__}-{num_iter}-{date_and_time}.pkl"

    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if attack == "GN":
            atk = attack(model, std=np.sqrt(eps))
            adv_examples = atk(X, y)
        elif is_torchattacks:
            atk = attack(model, eps=eps)
            adv_examples = atk(X, y)
        else:
            adv_examples = attack(model, loss_fn, (X, y), eps=eps, num_iter=num_iter) # (Ben) last parameter is for iterated FGSM.
        with open(pickle_file_path, 'wb+') as f:
            pickle.dump(adv_examples, f)

        pred = model(adv_examples)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>.3f}%, Avg loss: {test_loss:>8f} \n")

    with open(pickle_file_path, 'wb') as f:
        pickle.dump((correct, test_loss), f)

    # Save best vesion of model at save_path
    if(100*correct > best_acc) and save_best_model:
        print("Saving New Best Model")
        best_acc = 100*correct
        if model_path:
            save_path = model_path
        else: 
            save_path = BEST_MODEL_PATH 
        torch.save(model.state_dict(), save_path)
    

    return correct, test_loss


def train_with_adv(dataloader, model, loss_fn, optimizer, attack, eps=0.0, is_torchattacks=False, num_iter=1):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Adversarial perturbation using PGD
        adversary = attack(model, loss_fn, (X, y), eps=eps, num_iter=num_iter)

        # Compute prediction error
        pred = model(adversary)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, save_best_model=False, best_acc=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()        
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")

    # Save best vesion of model at save_path
    if(100*correct > best_acc) and save_best_model:
        print("Saving New Best Model")
        best_acc = 100*correct
        save_path = BEST_MODEL_PATH
        torch.save(model.state_dict(), save_path)
    
    return best_acc


# Set optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


## Loading the model with the best accuracy after a number of fine-tuning epochs
ADVERSARIAL_TRAINING_EPOCHS = 10
model_path = BEST_MODEL_PATH

for attack in [PGD, ifgsm]:
    print(f"--- Attack: {attack.__name__} ---")
    # Dataloaders, may need to change # of workers or batchsize to improve performance
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")
    model = model.to(DEVICE)
    test(test_loader, model, loss_fn, save_best_model=False) # Should be ~80 accuracy%

    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # attack = PGD # try with ifgsm as well

    epoch_results = []
    best_acc = 0
    eps = 0.01
    num_iter = 1
    for epoch in range(ADVERSARIAL_TRAINING_EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_with_adv(train_loader, model, loss_fn, optimizer, attack, eps=eps, is_torchattacks=False, num_iter=num_iter)
        best_acc, loss = test_with_adv(test_loader, model, loss_fn, attack, eps=eps, num_iter=num_iter, is_torchattacks=False, 
                    save_best_model=True, best_acc=best_acc, model_path=f"results/{attack.__name__}.pth")
        
        # Save results for this epoch
        epoch_results.append({
            "epoch": epoch + 1,  # Epoch is 1-indexed
            "best_acc": best_acc,
            "loss": loss
        })

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(epoch_results)
    df.to_csv(f"results/experiments/adversarial_training_{attack.__name__}.csv", index=False)

    print("Results for each epoch saved to epoch_results.csv")