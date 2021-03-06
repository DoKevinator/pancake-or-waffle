# pancake-or-waffle
Machine learning project to determine if pictures contain pancakes or waffles.

Uses Pytorch as the machine learning framework. 

This mini project is more of a little exercise to get me into the machine learning core concepts. 

HEAVILY based off of https://github.com/mlatsjsu/workshop-chihuahua-vs-muffin.

Synopsis:
I learned a lot from studying the chihuahua vs muffins project. Pytorch has tons of little technical aspects running under the hood that I currently don't even understand at all. I have lots to learn, but this is a good first step. I wonder if after training a model, you can save the model and run it in realtime? I would like to someday do realtime video tracking using Pytorch. In theory the training part is the most taxing part of the program? Running predictions after it's finished training shouldn't be too expensive so running it in realtime may be a reality. I'll have to do more research on what the profiling looks like for neural networks.
The final results aren't perfect. In fact, there's about a 75% hit rate for the pancakes vs waffles. I believe that if I can supply more data, then it would get more accurate. I also believe using more than 2 layers in the neural network is useless. I did some research and found that most applications only use up to 2 layers.

Applications:
In theory, this could be modified to identify any number of objects. It would just need to be scaled properly. All the concepts in here can be used to translate to other aspects. If I wanted to the program to distinguish between a sedan, truck, and semitruck, it can easily be done. 

![image](https://user-images.githubusercontent.com/4328910/115097069-e82a1d80-9edc-11eb-9904-d1cb2fc26816.png)
