# 🚀 DEEP CTF. CTF Team Name Generator

So, you're looking for a name for your CTF team? You've come to the right place! This is an evening project I did to generate some names for my team, however I thought it might be useful for others as well. So here it is! 🔥

## How to use

1. Clone the repo: `git clone https://github.com/m4drat/deep-ctf`
2. Train the model: `python3 deep_ctf.py --train`
3. Or just generate some names using the pre-trained model: `python3 deep_ctf.py --generate --num_names 10`

## How it works

The model is a simple GRU network with 2 hidden layers, trained on a dataset of 30'000+ CTF team names. The number of training epochs is around 60 in total, and the model was trained in 3 stages.

Some examples of the generated names:

```txt
h4ck3r_p0w3r
noobsquad
zer0team
shell rooot
chickensec
r4nd0m_m4n_t3am
cyb3rshell
team_1337
n00bst3am
cyber_shadow
```

In this repo you also can find a `dataset_generator.py` script, which was used to generate the dataset. It parses CTF team names from [ctftime.org](https://ctftime.org/teams/) and generates a dataset of 30'000+ team names. Alredy generated dataset is in `teams.txt` file.

### Training accuracy over 20 epochs

<!-- ![Training accuracy over 20 epochs](name_generator-acc-20-0.001-256-20-0.29112133383750916-v4.png) -->

<p align="left">
    <img src="name_generator-acc-20-0.001-256-20-0.29112133383750916-v4.png" width="75%" />
</p>

### Loss value over 20 epochs

<!-- ![Loss value over 20 epochs](name_generator-loss-20-0.001-256-20-0.29112133383750916-v4.png) -->

<p align="left">
    <img src="name_generator-loss-20-0.001-256-20-0.29112133383750916-v4.png" width="75%" />
</p>
