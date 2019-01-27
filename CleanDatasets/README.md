# Clean Datasets

Here, we randomly select clean samples that are correctly predicted by the corresponding model from the testing set of each dataset (i.e., MNIST and CIFAR10).

In addition, for target attacks, the target class is chosen randomly among the labels except the ground truth class.

Then, these selected clean samples will be attacked by all kinds of adversarial attacks.


```
python CandidatesSelection.py --dataset=MNIST/CIFAR10 --number=1000
```