
#### To generate a single entity file with all the manually labeled individual entity files

##### For bio space
  ./gen_bs.sh bio_list.txt
  
##### For PHI space
  ./gen_bs.sh phi_list.txt
  
  
  
Then use the output  bootstrap_entities.txt to cluster using the [bert_vector_clustering](https://github.com/ajitrajasekharan/bert_vector_clustering.git) . That will magnify the number of vocab terms tagged as well as generate a entity signature for all terms in vocab


#### Additional notes

This current set of manual labeling is still inadeqaute and there is scope for improved labeling. Specifically
  - for every entity type the adjectives that precede the entity needs to be labeled. This is hard to label manually given they can appear in the context of many entities. For example "chronic" is clearly an adjective that suggests it is a disease adjective. However, "standard" could be an adjective for any entity type. There are intermediates between this extreme. While one can label "chronic" easily, it gets harder as we move towards the generic adjective "standard". It may not even make sense to move close to "standard". However we could use the output of model for test sentences to tag the adjectives closer to the "chronic" end of the spectrum - a form of semi supervised labeling of terms assisted in part by the model output for those sentences. This could boost performance.
  - there are certain entity types, specifically in ENT (entertainment domain) that are very inadequately tagged
