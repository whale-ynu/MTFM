# MTFM
The pytorch implementation of each baseline model in the paper 《Mashup-Oriented Web API Recommendation based on Multi-Model Fusion and Multi-Task Learning》

## The code directory is as follows
* data/
  * api_category.json
  * api_description.json
  * api_name.json
  * api_quality_feature.dat
  * category_list.json
  * mashup_description.json
  * mashup_category.json
  * mashup_name.json
  * mashup_used_api.json
  * used_api_list.json
* model/
  * CF.python
  * NCF.py
  * SingleBPR.py
  * LSTM.py
  * Pop.py
  * FC-LSTM.py
  * RWR.py
  * SPR.py
  * MTFM.py
  * MTFM++.py
* tools/
  * utils.py
  * dataset_class.py
  * metric.py

Under the data folder is the data set file that we obtained from programmableweb.com and organized it.

Under the model folder is the implementation file of the baseline model involved in the paper.

Under the tools folder are the utils.py file for data preprocessing, the dataset_class.py file loaded by the dataset, and the metric.py file for metric calculation.
