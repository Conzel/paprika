# paprika
Paprika is an art project for the Museum of Tübingen, where a team of students visualize how a neural image classifer 
works. 

It is on display from 02/2023 - 10/2023 (https://www.tuebingen.de/stadtmuseum/38998.html). 

## Installation instructions
First download poetry. Then, with poetry install, switch to the directory root. Activate a shell using 

```poetry shell```

Then, install paprika using

```poetry install```

If you use VSCode, to get the venv to be picked up, type 

```poetry config virtualenvs.in-project true```

Before creating the venv. If you have already created the venv, delete it:

```poetry env list  
poetry env remove <current environment>
```

And then install the venv again.

## Repository structure
- `scripts` contains everything that is directly executable and serves some testing purpose
- `paprika` is the main module and contains three submodules: `ai`, `cam` and `ui`
- In each submodule, module's exports gets explicitly reimported in __init__.py. Module-internal files get prefaced with an
underscore

## Contributor Reminders 
- Format all code with `black` before commiting
- Use types in Python code where possible, check typing errors with `pyright`
- Add your dependencies to poetry, using `poetry add` or by hand
- Don't drop files into the repository root, use appropriate sub folders
- Write basic documentation where possible
- Write tests (under a `test` folder) if possible
- If you change code, see if the tests still run
- Write sensible commit messages that precisely explain your changes
- Write commits that serve one purpose (f.e. don't mix up formatting/refactoring and adding features)
- Develop larger features on a separate branch. Always leave the main branch in an intact state 
