# Installation
----
## Installing from source
```
(准备 Python 3.13 的 conda 环境)
conda create --name AL-pipeline python=3.13
conda activate AL-pipeline
pip install uv
git clone git@github.com:cellethology/AL-pipe.git
cd AL-pipe
uv sync --dev
(同步依赖（含开发依赖）)
```

# TODO 
------
- [ ] Double check the classes in python is in adherence to the public and private syntax
- [ ] Beware of the `# @package _global_` namespace in hydra yaml config
- [ ] Write plotting function integrated with the trainer
- [ ] Check all the todos
- [ ] Update the codespace with datamodule class (easier to manage data) `trainer.test(model, datamodule=dm)` 
- [ ] Double check if the seed is correct
