
#### To generate a single entity file with all the manually labeled individual entity files

##### For bio space
  ./gen_bs.sh bio_list.txt
  
##### For PHI space
  ./gen_bs.sh phi_list.txt
  
  
  
Then use the output  bootstrap_entities.txt to cluster using the [bert_vector_clustering](https://github.com/ajitrajasekharan/bert_vector_clustering.git) . That will magnify the number of vocab terms tagged as well as generate a entity signature for all terms in vocab
