# Spelling Correction


## About The Project


### Introduction

My project is called **Spelling Correction**. This is a solution which helps a person correct a sentence with spelling mistakes. This repository contains 2 versions of my soluion. The first one is successfully built (from scratch) with Transformer architecture. Besides, the second one is fine-tuned with GPT-2. However, unluckily, the second one is completely coded but it has not been trained because my computational resources were limited.

## Getting Started

To get started, you should have prior knowledge on **Python** and **Pytorch** at first. A few resources to get you started if this is your first Python or Tensorflow project:

- [Pytorch Tutorials](https://pytorch.org/tutorials/)
- [Python for Beginners](https://www.python.org/about/gettingstarted/)


## Installation and Run
1. Clone the repo

   ```sh
   git clone https://github.com/phkhanhtrinh23/spelling_correction_project.git
   ```
  
2. Use any code editor to open the folder **spelling_correction_project**. With `python=3.8`, run `pip install -r requirements.txt` in your corresponding conda venv.

### Version 1
3. Download the [weights](https://drive.google.com/drive/folders/19r7GYrIvAVtZWhJ_mDchJWWnbLMCvXZi?usp=sharing) into `spelling_correction_v1`.

4. Download the data [`english.txt`](https://drive.google.com/file/d/1uVAKtFW5OXJMO1clhF_vDZYVwWa_FDTL/view?usp=sharing) into the `data/` folder in `spelling_correction_v1`. The correct path is `data/english.txt`.

5. Run `python train.py` to train the model using Transformer architecture.

6. Run `python api.py` to run the Front-end + Back-end Web Demo for this application. Share your results with me!

### Version 2
3. Download the data [`english.txt`](https://drive.google.com/file/d/1uVAKtFW5OXJMO1clhF_vDZYVwWa_FDTL/view?usp=sharing) into the `data/` folder in `spelling_correction_v1`. The correct path is `data/english.txt`.

4. Run `python train.py` to train the model using GPT-2.

5. The log is saved in `logs/` folder.

6. If the training is finised, you can run `python evaluate.py` to evaluate the results. Again, share your results with me if possible!


## Outline

- Input: [`english.txt`](http://www.manythings.org/anki/) a English Dictionary.

- Output:
   - `spelling_correction_v1` is based on **Transformer Encoder-Decoder** model. It is fast in training and inference.
   - `spelling_correction_v2` is based on **GPT-2** from [huggingface.co](https://huggingface.co/gpt2-medium).

## Results
- This is the result from the successfully built `spelling_correction_v1`.

<img src="./spelling_correction_v1/images/output.png"/>

- Link to the [YouTube demonstration](https://youtu.be/J-JnNqeN9zU).


## Contribution

Contributions are what make GitHub such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the project
2. Create your Contribute branch: `git checkout -b contribute/Contribute`
3. Commit your changes: `git commit -m 'add your messages'`
4. Push to the branch: `git push origin contribute/Contribute`
5. Open a pull request


## Contact

Email: phkhanhtrinh23@gmail.com