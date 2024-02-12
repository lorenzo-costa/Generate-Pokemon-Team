## Generate Pokemon Team (GPT)
The Pokemon franchise has marked my childhood (and not only) so to test my skills and learn something new I decided to create a model that finds new names (Nintendo you're welcome).
The model is a transformed-based neural network following the architecture presented in the landmark paper [Attention is All You Need](https://doi.org/10.48550/arXiv.1706.03762) and was inspired by the amazing lecture by [Andrej Karpathy](https://en.wikipedia.org/wiki/Andrej_Karpathy) available for free on YouTube ([here](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)).

More specifically:
- the model_definition.py file contains the model definition (duh?)
- training was done in two steps: first 15k iterations on a dataset containing the 30k most common US names taken from [ssa.gov](https://www.ssa.gov/oact/babynames/)  and then 800 "fine-tuning" iterations on a dataset containing all (real) Pokemon names. The idea behind this two step training is that there are only ~900 unique Pokemon names, which are not enough to fully train the network. By pre-training the network on real (human) names the model is better able to learn, while avoiding overfitting.
- the file pkm_run.ipynb demonstrates the functioning of the model.

Is this architecture overkill for this task? Most likely.\
Was it fun and worth the hussle? YES

Example of a team GPT was able to generate are:
```
Yuroaba  
Glassoon 
Endigoon
Selcon 
Mantle 
Cangrost
```
p.s. Mantle (beside being an english word) is an actual character from the Pokemon cartoon. The model has then generate, totally by itself because Mantle is not in the trainig data, a real name. How cool is that!

To complete this I was curious of how these would look like. This is what Dall-E thinks:
![Picture1](https://github.com/lorenzo-costa/Generate-Pokemon-Team/assets/149969774/d143ad1f-b1b7-476f-a70f-59b19df63150)

