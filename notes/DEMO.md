# Demo

## Setup

* CRISP-T is a python package that can be installed using pip and used from the command line. Your system should have python 3.11 or higher and pip installed. You can download and install Python for your operating system [here](https://www.python.org/downloads/). Optionally, CRISP-T can be imported in python scripts or jupyter notebooks, but this is not covered in this demo. See [documentation](https://dermatologist.github.io/crisp-t/) for more details.
* Install [Crisp-T](https://github.com/dermatologist/crisp-t) with `pip install crisp-t[ml]` or `uv pip install crisp-t[ml]`
* (Optional) Download covid narratives data to  `crisp_source` folder in home directory or current directory using `crisp --covid covidstories.omeka.net --source crisp_source`. You may use any other source of textual data (e.g. journal articles, interview transcripts) in .txt or .pdf format in the `crisp_source` folder or the folder you specify with --source option.
* (Optional) Download [Psycological Effects of COVID](https://www.kaggle.com/datasets/hemanthhari/psycological-effects-of-covid) dataset to `crisp_source` folder. You may use any other numeric dataset in .csv format in the `crisp_source` folder or the folder you specify with --source option.
* Create a `crisp_input` folder in home directory or current directory for keeping imported data for analysis.

## Import data

* Run the following command to import data from `crisp_source` folder to `crisp_input` folder.
* `--source` reads data from a directory (reads .txt, .pdf and a single .csv) or from a URL

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
crisp --print documents
```

* Notice that we have omitted --inp as it defaults to `crisp_input` folder. If you have a different folder, use --inp to specify it.
* Notice keywords assigned to each narrative.
* You will notice *interviewee* and *interviewer* keywords. These are assigned based on the presence of these words in the narratives and may not be useful.
* You may remove these keywords by using --ignore with assign and check the results again.

```bash
crisp --out crisp_input --assign --ignore interviewee,interviewer
crisp --print documents
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

## Now let us try out a csv dataset with text and numeric data.

* Download SMS Smishing Collection Data Set from [Kaggle](https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set) and convert the text file to csv adding the headers id, **CLASS** and **SMS**. Convert CLASS to numeric 0 and 1 for ham and smish respectively and add id as serial numbers.
* Place the csv file in a **new** `crisp_source` folder.
* Import the csv file to `crisp_input` folder using the following command.

```bash
crisp --source crisp_source/ --out crisp_input --unstructured SMS
```

* Notice that the text column SMS is specified with --unstructured option. This creates CRISP documents from the text column.
* Now assign topics to the documents. Note that this also assigns clusters.

```bash
crisp --inp crisp_input/ --out crisp_input/ --assign
```

* Now print the results to examine.
```bash
crisp --print documents
```

* Let us choose the cluster 1 and see the SMS classes in this cluster. (0=ham, 1=smish)
```bash
crisp --filters cluster=1 --print stats
```

* Next, let us check if the SMS texts converge towards predicting the CLASS (ham/ smish) variable with LSTM model.

```bash
crisp --lstm --outcome CLASS
```

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

## Visualization


### Let's [visualize the clusters in 2D space using PCA.](https://htmlpreview.github.io/?https://github.com/dermatologist/crisp-t/blob/develop/notes/lda_visualization.html)

```bash
crispviz --inp crisp_input/ --out crisp_output/ --ldavis
```

* The visualization will be saved in `crisp_output` folder. Open the html file in a browser to explore.

### Let's generate a word cloud of keywords in the corpus.

```bash
crispviz --inp crisp_input/ --out crisp_output/ --wordcloud
```
* The word cloud will be saved in `crisp_output` folder.

<p align="center">
  <img src="https://github.com/dermatologist/crisp-t/blob/develop/notes/wordcloud.jpg" />
</p>