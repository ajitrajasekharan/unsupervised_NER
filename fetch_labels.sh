mkdir -p labels 
( cd labels;
gdown https://drive.google.com/uc?id=1QmJBoLsKFqU8X_Q0sGWA1uyYsyqH7lOB
tar xvf labels.tar
mv labels/* .
rm -rf labels
gdown https://drive.google.com/uc?id=19E51jqh3ZQG6FayibJIcnyUOBqnFTLAl
tar xvf server_config.tar
)
