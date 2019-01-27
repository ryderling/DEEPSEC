# Attacks

Implementations of adversarial attack algorithms that are used to generate adversarial examples.

## Description

For each attack, we first define and implement the attack class (e.g., **`FGSMAttack`** within **`FGSM.py`** for the FGSM attack) in **`Attacks/AttackMethods/`** folder, then we implement the testing code (e.g., **`FGSM_Generation.py`**) to generate corresponding adversarial examples and save them into the directory of [`AdversarialExampleDatasets/`](../AdversarialExampleDatasets/). Therefore, you can generate any adversarial examples you want by specifying their parameters accordingly.

## Implemented Attacks
Here, we implement 16 state-of-the-art adversarial attacks, including 8 un-targeted attack and 8 targeted attack.

- [x] **FGSM**: *I. J. Goodfellow, et al., "Explaining and harnessing adversarial examples," in ICLR, 2015.*
- [x] **R+FGSM**: *F. Tram`er, et al., "Ensemble adversarial training: Attacks and defenses," in ICLR, 2018.*
- [x] **BIM**: *A. Kurakin, et al., "Adversarial examples in the physical world," in ICLR, 2017.*
- [x] **PGD**: *A. Madry, et al., "Towards deep learning models resistant to adversarial attacks," in ICLR, 2018.*
- [x] **U-MI-FGSM**: *Y. Dong, et al., "Boosting adversarial attacks with momentum," arXiv:1710.06081, 2017.*
- [x] **DF**: *S.-M. Moosavi-Dezfooli, et al., "Deepfool: A simple and accurate method to fool deep neural networks," in CVPR, 2016.*
- [x] **UAP**: *S.-M. Moosavi-Dezfooli, et al., "Universal adversarial perturbations," in CVPR, 2017.*
- [x] **OM**: *W. He, et al., "Decision boundary analysis of adversarial examples," in ICLR, 2018.*

- [x] **LLC**: *A. Kurakin, et al., "Adversarial examples in the physical world," in ICLR, 2017.*
- [x] **R+LLC**: *F. Tram`er, et al., "Ensemble adversarial training: Attacks and defenses," in ICLR, 2018.*
- [x] **ILLC**: *A. Kurakin, et al., "Adversarial examples in the physical world," in ICLR, 2017.*
- [x] **T-MI-FGSM**: *Y. Dong, et al., "Boosting adversarial attacks with momentum," arXiv:1710.06081, 2017.*
- [x] **BLB**: *C. Szegedy, et al., "Intriguing properties of neural networks," in ICLR, 2014.*
- [x] **JSMA**: *N. Papernot, et al., "The limitations of deep learning in adversarial settings," in EuroS&P, 2016.*
- [x] **CW2**: *N. Carlini and D. Wagner, "Towards evaluating the robustness of neural networks," in S&P, 2017.*
- [x] **EAD**: *P. Chen, et al., "EAD: elastic-net attacks to deep neural networks via adversarial examples," in AAAI, 2018.*


## Usage

Generation of adversarial examples with specific attacking parameters that we used in our evaluation.

|   Attacks   | Commands with default parameters |
|:-----------:|--------------------------------- |
| **FGSM**    | python FGSM_Generation.py --dataset=MNIST --epsilon=0.3 <br> python FGSM_Generation.py --dataset=CIFAR10 --epsilon=0.1|
| **RFGSM**   | python RFGSM_Generation.py --dataset=MNIST --epsilon=0.3 --alpha=0.5 <br> python RFGSM_Generation.py --dataset=CIFAR10 --epsilon=0.1 --alpha=0.5 |
| **BIM**     | python BIM_Generation.py --dataset=MNIST --epsilon=0.3 --epsilon_iter=0.05 --num_steps=15 <br> python BIM_Generation.py --dataset=CIFAR10  --epsilon=0.1 --epsilon_iter=0.01 --num_steps=15 |
| **PGD**     | python PGD_Generation.py --dataset=MNIST --epsilon=0.3 --epsilon_iter=0.05 <br> python PGD_Generation.py --dataset=CIFAR10 --epsilon=0.1 --epsilon_iter=0.01 |
| **UMIFGSM** | python UMIFGSM_Generation.py --dataset=MNIST --epsilon=0.3 --epsilon_iter=0.05 <br> python UMIFGSM_Generation.py --dataset=CIFAR10 --epsilon=0.1 --epsilon_iter=0.01 |
| **UAP**     | python UAP_Generation.py --dataset=MNIST --fool_rate=0.35 --epsilon=0.3 <br> python UAP_Generation.py --dataset=CIFAR10 --fool_rate=0.9 --epsilon=0.1 |
| **DeepFool**| python DeepFool_Generation.py --dataset=MNIST --max_iters=50 --overshoot=0.02 <br> python DeepFool_Generation.py --dataset=CIFAR10 --max_iters=50 --overshoot=0.02 |
| **OM**      | python OM_Generation.py --dataset=MNIST --initial_const=0.02 --learning_rate=0.2 --noise_count=20 --noise_mag=0.3 <br> python OM_Generation.py --dataset=CIFAR10 --initial_const=1 --learning_rate=0.02 --noise_count=20 --noise_mag=0.03137255 |
| **LLC**     | python LLC_Generation.py --dataset=MNIST --epsilon=0.3 <br> python LLC_Generation.py --dataset=CIFAR10 --epsilon=0.1 |
| **RLLC**    | python RLLC_Generation.py --dataset=MNIST --epsilon=0.3 --alpha=0.5 <br> python RLLC_Generation.py --dataset=CIFAR10 --epsilon=0.1 --alpha=0.5 |
| **ILLC**    | python ILLC_Generation.py --dataset=MNIST --epsilon=0.3 --epsilon_iter=0.05 <br> python ILLC_Generation.py --dataset=CIFAR10 --epsilon=0.1 --epsilon_iter=0.01 |
| **TMIFGSM** | python TMIFGSM_Generation.py --dataset=MNIST --epsilon=0.3 --epsilon_iter=0.05 <br> python TMIFGSM_Generation.py --dataset=CIFAR10 --epsilon=0.1 --epsilon_iter=0.01 |
| **JSMA**    | python JSMA_Generation.py --dataset=MNIST --theta=1.0 --gamma=0.1 <br> python JSMA_Generation.py --dataset=CIFAR10 --theta=1.0 --gamma=0.1 |
| **BLB**     | python BLB_Generation.py --dataset=MNIST <br> python BLB_Generation.py --dataset=CIFAR10 |
| **CW2**     | python CW2_Generation.py --dataset=MNIST --confidence=0 --initial_const=0.001 <br> python CW2_Generation.py --dataset=CIFAR10 --confidence=0 --initial_const=0.001 |
| **EAD**     | python EAD_Generation.py --dataset=MNIST --confidence=0 --beta=0.001 --EN=True <br> python EAD_Generation.py --dataset=CIFAR10 --confidence=0 --beta=0.001 --EN=True |
