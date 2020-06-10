# Unsupervised NER (prototype)
Prototype unsupervised NER using BERT's MLM and wrapper around [Dat Quoc Nguyen's POS tagger/Dependency parser](https://github.com/datquocnguyen/jPTDP)


[Medium post describing this method](https://towardsdatascience.com/unsupervised-ner-using-bert-2d7af5f90b8a)

![Image from Medium post on unsupervised NER ](NER.png)


# Installation 

**1) Install POS service using https://github.com/ajitrajasekharan/JPTDP_wrapper.git**

Confirm installation works by 

    $ wget -O POS "http://127.0.0.1:8073/John flew from New York to Rio De Janiro"
    
   The output POS file should contain
  
  ![The output POS file should contain ](POS.png)
  
 
 2) **Install BERT descriptor service using https://github.com/ajitrajasekharan/bert_descriptors.git**
 
 Confirm installation works by 
 
   $ wget -O DES "http://127.0.0.1:8087/dummy/John flew from entity to Rio De Janiro"
   
   The output DES file should contain
   
 ![The output POS file should contain ](DES.png)
  

# License

This repository is covered by MIT license. 

[The POS tagger/Dep parser  is covered by a GPL license.](https://github.com/datquocnguyen/jPTDP/blob/master/License.txt)
