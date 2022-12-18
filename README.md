## Removing Redundant Test Cases That Are Specified in Natural Language

This repository contains the source code of our course project MM811 wherein we are supposed to remove similar test cases that are specified in natural language.


</br>
In today’s rapidly advancing software industry,  we experience exciting technological growth every year in an extensive range of fields such as innovative AI-powered development, cloud and edge computing, machine learning, progressive and lightweight web and mobile applications etc. However, it is not quite relevant to the Software Testing industry. Most companies still rely on the outdated manual testing process despite the availability of Automation testing procedures. It may work for small teams testing the software using a limited number of test cases. Although, with the increase in team size and the number of test cases over time, the cost, effort and time needed to manually validate and review the test cases increase tenfold. It often leads to recurrent and unclear test cases in the test suite, delivered by different employees who often work across teams. These test cases are represented in Natural Language. Also, the redundant test cases can impact the manual testing process by testing the same feature multiple times and can reduce the possibility of writing multiple methods to test the same feature when automating tests in the future. Hence, in this project, we propose to address the problem of similar test cases using an unsupervised learning approach. Additionally, after removing redundancy in the test suite, we intend to identify key features in the software to be tested based on the description of the test cases in the suite, and group multiple test cases into a software feature. This feature can be directly assigned to a Quality Assurance engineer for testing. 

## Techniques Used

To cluster similar test steps, we performed several experiments with the following text embedding techniques, text similarity metrics, and clustering algorithms:

**Text embedding techniques**

* [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf)


**Text similarity metrics**

* [Word  Mover’s  Distance  (WMD) ](http://proceedings.mlr.press/v37/kusnerb15.pdf)


**Clustering algorithms**

* [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)



## Steps to execute the code:
- Step-I: Test Steps clustering with Word2Vec
  - Open the Jupyter Notebook file *[I_Test_Step_Clustering_Word2Vec.ipynb](I_Test_Step_Clustering_Word2Vec.ipynb)* and execute all the cells one-by-one.
  - Run the *Code Executon* cell block.


- Step-II: Ensemble Technique using Majority Voting Approach
  - Based on the clustered data and clustered label files we get we use it in the ensemble approach.
  - Open the Jupyter Notebook file *[II_EnsembleClustering.ipynb](II_EnsembleClustering.ipynb)* and execute all the cells one-by-one.
  
- Step-III: Cosine Similarity for Test Cases and Test Steps
  - From Step-II we will be getting the pairwise similar distances of the test steps and test cases and will be using in this step.
  - Open the Jupyter Notebook file *[III_CosineSimilarity.ipynb](III_CosineSimilarity.ipynb)* and execute all the cells one-by-one.
  
- Step-IV: Final Output Test Case File
  - After getting the similarity score of all the test steps we use the .txt file generated from the code to generate a new output test cases file which will not have the redundant test cases and steps
  - Open the Jupyter Notebook file *[IV_FinalConversion.ipynb](IV_FinalConversion.ipynb)* and execute all the cells one-by-one.



## Implementation:
- Loads the training dataset provided titled *[test_cases.xlsx](test_cases.xlsx)*
- Performs pre-processing opeartions on the training dataset comprising of the functions *get_unique_word_count*, *get_word_frequency*, *read_input_data*, *read_input_data* and *return_training_list*
- Next, a pre-trained Word2Vec model is additionally trained with the dataset. This i


## Citations and References:
- For understanding of our project requirements and the necessary steps required to achieve the objective, we extensively referred the research paper [Identifying similar test cases that are specified in natural language](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9763328).
- For the code implementation, we referred the repository [asgaardlab/21-markos-test_case_similarity_technique-code](https://github.com/asgaardlab/21-markos-test_case_similarity_technique-code).
- Citations have been added in *[bibliography.bib](bibliography.bib)*
    
