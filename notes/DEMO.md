# Demo

## Setup

* Install [Crisp-T](https://github.com/dermatologist/crisp-t) with `pip install crisp-t[ml]` or `uv pip install crisp-t[ml]`
* Download covid narratives data to  `crisp_source` folder in home directory or current directory using `crisp --covid covidstories.omeka.net --source crisp_source`
* Create a `crisp_input` folder in home directory or current directory for keeping imported data.
* Download [Psycological Effects of COVID](https://www.kaggle.com/datasets/hemanthhari/psycological-effects-of-covid) dataset to `crisp_source` folder.

## Import data

* Run the following command to import data from `crisp_source` folder to `crisp_input` folder.
```bash
crisp --source crisp_source --out crisp_input
```
* Ignore warnings related to pdf files.

## Perform Exploratory tasks using NLP

* Run the following command to perform a topic modelling and assign topics(keywords) to each narrative.

```bash
crisp --inp crisp_input --out crisp_input --assign
```

* The results will be saved in the same `crisp_input` folder, overwriting the corpus file.
* You may run several other analyses ([see documentation](https://dermatologist.github.io/crisp-t/) for details) and tweak parameters as needed.
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

## Quantitative exploratory analysis

* Let us see do a kmeans clustering of the csv dataset of covid data.

```bash
crisp --include relaxed,self_time,sleep_bal,time_dp,travel_time,home_env --kmeans
```

* Notice 3 clusters with different centroids. (number of clusters can be changed with --num option). Profile of each cluster can be seen with --profile option.

## Confirmation

* Let us add a relationship between numb:self_time and text:work in the corpus for future confirmation with LLMs.

```bash
crispt --add-rel "text:work|numb:self_time|correlates" --out crisp_input
```

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

## [Sense-making by triangulation](INSTRUCTION.md)

## MCP Server for agentic AI. (Optional, but LLMs may be better at sense-making!)

### Try out the MCP server with the following command. (LLMs will offer course corrections and suggestions)


* load corpus from /Users/your-user-id/crisp_input
* use available tools
* What are the columns in df?
* Do a regression using time_bp,time_dp,travel_time,self_time with relaxed as outcome
* Interpret the results
* Is self_time or related concepts occur frequently in documents?
* can you ignore "interviewer,interviewee" and assign topics again? Yes.
* What are the topics in documents with keyword "work"?

<p align="center">
  <img src="https://github.com/dermatologist/crisp-t/blob/develop/notes/crisp.gif" />
</p>
