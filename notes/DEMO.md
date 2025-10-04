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

* Run the following command to perform a topic modelling and assign topics(keywords) to each narrative.
```bash
crisp --inp crisp_input --out crisp_input --assign
```
* The results will be saved in the same `crisp_input` folder, overwriting the corpus file.
* You may run several other analyses and tweak parameters as needed.
* Hints will be provided in the terminal.

## Explore results

```bash
crisp -- print
```
* Notice that we have omitted --inp as it defaults to `crisp_input` folder. If you have a different folder, use --inp to specify it.
* Notice keywords assigned to each narrative.
* You will notice *interviewee* and *interviewer* keywords. These are assigned based on the presence of these words in the narratives and may not be useful.
* You may remove these keywords by using --ignore with assign and check the results again.

```bash
crisp --out crisp_input --assign --ignore interviewee,interviewer
crisp -- print
```
* Now you will see that these keywords are removed from the results.
* Let us choose narratives that contain 'work' keyword and show the concepts/topics in these narratives.
```bash
crisp --filters keywords=work --topics
```

* `Applied filters ['keywords=work']; remaining documents: 51`
* Notice *time*, *people* as topics in this subset of narratives.

## Quantitative analysis

* Let us see do a kmeans clustering of the csv dataset of covid data.
```bash
crisp --include relaxed,self_time,sleep_bal,time_dp,travel_time,home_env --kmeans
```
* Notice 3 clusters with different centroids. (number of clusters can be changed with --num option)
* Let us do a regression analysis to see how `relaxed` is affected by other variables.
```bash
crisp --include relaxed,self_time,sleep_bal,time_dp,travel_time,home_env --regression --outcome relaxed
```
* self_time has a positive correlation with relaxed.
* What about a decision tree analysis?
```bash
crisp --include relaxed,self_time,sleep_bal,time_dp,travel_time,home_env --cls --outcome relaxed
```
* Notice that self_time is the most important variable in predicting relaxed.

## Confirmation

* Let us add a relationship between numb:self_time and text:work in the corpus for future confirmation with LLMs.
```bash
crispt --add-rel "text:work|numb:self_time|correlates" --out crisp_input
```

## [TRIANGULATION](INSTRUCTION.md)


## MCP Server for agentic AI
