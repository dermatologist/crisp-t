# Demo

* Install [Crisp-T](https://github.com/dermatologist/crisp-t) with `pip install crisp-t` or `uv pip install crisp-t`
* move covid narratives data to  `crisp_source` folder in home directory or current directory.
* create a `crisp_input` folder in home directory or current directory for keeping imported data.
* copy [Psycological Effects of COVID](https://www.kaggle.com/datasets/hemanthhari/psycological-effects-of-covid) dataset to `crisp_source` folder.

## Import data

* Run the following command to import data from `crisp_source` folder to `crisp_input` folder.
```bash
crisp --source crisp_source --out crisp_input
```
* Ignore warnings related to pdf files.

## Perform NLP tasks

* Run the following command to perform NLP tasks on the imported data.
```bash
crisp --inp crisp_input --out crisp_input --nlp
```
* The results will be saved in the same `crisp_input` folder, overwriting the corpus file.
* You may run individual analysis and tweak parameters as needed.
* Hints will be provided in the terminal.

## Explore results
