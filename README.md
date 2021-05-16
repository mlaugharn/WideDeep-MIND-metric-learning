First create a new conda environment and install the following packages: numpy, pytorch, pandas, sklearn, pytorch-widedeep, pytorch-metric-learning, annoy, faiss, IPython, tqdm

I had to tweak some of the source code of pytorch-metric-learning and pytorch-widedeep, so replace the following files in the new environment's Lib/site-packages/ folder with these files found in patched/

- pytorch_metric_learning\testers\base_tester.py
- pytorch_metric_learning\utils\inference.py
- pytorch_widedeep\training\trainer.py

To run the script:
1. python create_features.py
1. python wideanddeep.py

Once its done you will be presented with an IPython shell so you could still gather some data about the model and
predictions.

Also, currently the tweak I made in base_tester.py assumes you will be using a cuda device. If not, just use .cpu() where I put .cuda(), on line 136
