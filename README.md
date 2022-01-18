### Self-supervised NER (prototype)
_This repository containes code for solving NER  with self-supervised learning (SSL) alone avoiding supervised learning._
<br/>
<br/>
<br/>

 <img src="NER.png" width="600">



[Post describing the second iteration of this method](https://ajitrajasekharan.github.io/2021/01/02/my-first-post.html)

### Model performance on 11 datasets

<image src="performance.png" width="600">

### Additional links

- [Test repository link](https://github.com/ajitrajasekharan/ner_test.git) used to test this approach
- [Medium post describing the previous iteration of this method](https://towardsdatascience.com/unsupervised-ner-using-bert-2d7af5f90b8a)
- To identify noun phrase spans, [Dat Quoc Nguyen's POS tagger/Dependency parser](https://github.com/datquocnguyen/jPTDP) is used.

### Installation 
 
 <img src="ensemble.png" width="600">

 _If the use case is to automatically detect all noun phrase spans in a sentence, then POS tagger needs to be installed. If we only require specific phrases of interest to us in a sentence to be tagged (e.g. colorectal cancer above), then POS tagger install is **not** required. In the first use case, 7 microservices (POS tagger is made up of two microservices)  are started. In the second use, case 5 microservices are started._ 
 
### Step 1. Installing and starting microservices common to both use cases
 
 Run 
 ./setup.sh
 
 _this will install and load all 5 microservices. When done _(assuming all goes well)_ should display the output of a test query_

 ### Step 2. Install POS service 
 _**(this can be skipped if we only require specific phrases to be tagged)**_
 
 Install POS service using  [this link](https://github.com/ajitrajasekharan/JPTDP_wrapper.git)

*Make sure to run **both** services in the install instructions*

_Note POS service requires python 2.7 environment_
  
    
 ### Revision notes for major updates
 
 17 Jan 2022
 - Ensemble service of NER with two models tested on 11  NER benchmarks as described in this [post.](https://ajitrajasekharan.github.io/2021/01/02/my-first-post.html)
 
 
 17 Sept 2021
 
 - This can now be run as a service. run_servers.sh
 - Simple Ensembling service added for combining results of multiple NER servers
 
 
###  First Version Usage notes 

The unsupervised NER tool  can be used in three ways. 

1) to tag canned sentences (option 1)
     - $ python3 main_ner.py 1 
2) To tag custom sentences present in a file (option 2)
    - $ python3 main_ner.py 2 sample_test.txt
3) To tag single entities in custom sentences present in a file (option 3) where the single entity is specified in a sentence in the format name:__ entity __ . Concrete example: Cats and Dogs:__ entity __ are pets where Dogs is the term to be tagged. Single or multiple words/phrases within a sentence can also be tagged. Example: Her hypophysitis:__ entity __ secondary to ipilimumab:__ entity __ was well managed with supplemental:__ entity__ hormones:__ entity __
    - $ python main_NER.py 3 single_entity_test.txt
    
    
 
  

### License

This repository is covered by MIT license. 

[The POS tagger/Dep parser that this service depends on is covered by a GPL license.](https://github.com/datquocnguyen/jPTDP/blob/master/License.txt)
