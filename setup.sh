
function init
{
    echo "Installing basic requirements"
    pip install gdown
    pip install pytorch_pretrained_bert
    pip install pytorch_transformers
    pip install scikit-learn
    pip install transformers
}


function fetch
{
    echo "Fetching models"
    ./fetch_models.sh

    echo "Fetching entity vectors"
    ./fetch_labels.sh

}

function install_bert_desc
{
    param=$1
    git clone https://github.com/ajitrajasekharan/bert_descriptors.git
    if [ $param -eq 2 ]
    then
        echo "Starting BBC descriptor service"
        (cd bert_descriptors; cp ../../../models/bbc/* . ; cp ../../../labels/bbc_labels.txt ./labels.txt; cp ../../../labels/desc_bbc_config.json ./server_config.json;  echo "python p3_batch_server.py 8088" >  ./run_batched_response_servers.sh ; ./run_batched_response_servers.sh &)
    else
        echo "Starting BIO descriptor service"
        (cd bert_descriptors; cp ../../../models/a100/* . ;cp ../../../labels/a100_labels.txt ./labels.txt; cp ../../../labels/desc_a100_config.json ./server_config.json ;  ./run_batched_response_servers.sh &)
    fi
}

function install_ner
{
    param=$1
    git clone https://github.com/ajitrajasekharan/unsupervised_NER.git
    if [ $param -eq 2 ]
    then
        echo "Starting BBC NER service"
        (cd unsupervised_NER; cp ../../../labels/ner_bbc_config.json ./config.json; echo "python batched_p3_server.py 9089" >  ./batched_run_server.sh; ./batched_run_server.sh & )
    else
        echo "Starting BIO NER service"
        (cd unsupervised_NER; cp ../../../labels/ner_a100_config.json ./config.json; ./batched_run_server.sh &)
    fi
}

function microservices
{
    echo "Installing microservices"
    mkdir -p services
    (cd services;mkdir -p bio; cd bio; install_bert_desc 1; install_ner 1; )
    (cd services;mkdir -p phi;  cd phi; install_bert_desc 2; install_ner 2; )
    echo "Starting ensemble server"
    (cd ensemble; ./run_server_json.sh &)
    echo "Testing install with query"
    sleep 5
    wget -O test "http://127.0.0.1:8059/dummy/Lou:__entity__ Gehrig:__entity__ who works in XCorp:__entity__ suffers from Parkinson's:__entity__"
    echo "output saved in test"
    more test
    
}



fetch
microservices
