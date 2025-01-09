# Accuracy based XAI evaluation

The relevance of the saliency map highlights is evaluated by the classification model itself.

The change of accuracy in dependence of the revealed fraction is determined.



# Development notes:

`axaiev pg-b-mp --mask-dir /home/ck/mnt/XAI-DIA-gl/Julian/Dataset_Masterarbeit/atsds_large_ground_truth/train/`


# Contributing

We highly welcome external contributions. To reduce friction losses in a growing team we have the following guide lines.

## Code

- We (aim to) use `black -l 110 ./` to ensure coding style consistency, see also: [code style black](https://github.com/psf/black).
- We strongly encourage writing/updating doc strings
- We recommend using [typing hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- We strongly encourage to adopt a test-first-paradigm: write a (failing) unittest before the actual implementation.


## Git

- We loosely follow the [git flow branching model](https://nvie.com/posts/a-successful-git-branching-model/): New features should be developed either in their own branch or a a personal development-branch like `develop_ck`. From there they are merged into `develop` (pull requests should usually target this branch). The `main` branch is then updated as needed (e.g. by downstream dependencies).
- For commit messages we (mostly) follow the [conventional commits specification](https://www.conventionalcommits.org/en/).