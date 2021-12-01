# MTFM
The pytorch implementation of evaluated models in our work:

Hao Wu, Yunhao Duan, Kun Yue, Lei Zhang. Mashup-Oriented Web API Recommendation via Multi-Model Fusion and Multi-Task Learning. IEEE Transactions on Services Computing, 2021. DOI: 10.1109/TSC.2021.3098756

The data folder includes datasets obtained from programmableweb.com, most of them are packed into data files in the format of JSON.

The model folder contains the source codes of all baseline models as well as our proposed MTFM/MTFM++ models. 

The tools folder contains utils.py for data preprocessing,  dataset_class.py loaded by the dataset, and  metric.py for metric calculation.

Currently, there are four metrics known as MAP@N, NDCG@N, Precision@N and Recall@N, are used to evaluate the ranking performances of different recommendation models.
For the evaluation metric of compatibility (CPB@N) of the candidate set of APIs, please refer to our paper to find the definition.

## The code directory is as follows
* data/
  * api_category.json-- The relationship of Web APIs and Categories
  * api_description.json -- The textual documents of Web APIs
  * api_name.json -- The name string of Web  APIs
  * api_quality_feature.dat -- The naive quality vector of Web APIs 
  * category_list.json -- The mapping of category string to category identifier
  * mashup_description.json -- The textual documents of Mashups
  * mashup_category.json -- The relationship of Mashups and Categories
  * mashup_name.json -- The name string of Mashups
  * mashup_used_api.json -- The relationship of Mashups and used Web APIs
  * used_api_list.json-- The list of APIs used by each Mashup
* model/
  * CF.python -- Collaborative Filtering
  * NCF.py -- Neural Collaborative Filtering
  * SingleBPR.py -- Bayesian Personalized Ranking
  * LSTM.py -- LSTM Model
  * Pop.py -- Popularity-based model
  * FC-LSTM.py -- Functional and Contextual Attention-based LSTM model
  * RWR.py -- Random Walks with Restart
  * SPR.py -- Service Profile Reconstruction model
  * MTFM.py -- Multi-Task Fusion Model
  * MTFM++.py -- Extended MTFM to exploit the metadata and quality features of Web APIs
* tools/
  * utils.py
  * dataset_class.py
  * metric.py
