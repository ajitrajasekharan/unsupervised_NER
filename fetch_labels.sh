mkdir -p labels 
( cd labels;
gdown https://drive.google.com/uc?id=13YcDHsw2IavKbVba7vXICcLvVH1_5KWL
tar xvf labels.tar
mv labels/* .
rm -rf labels
gdown https://drive.google.com/uc?id=19E51jqh3ZQG6FayibJIcnyUOBqnFTLAl
tar xvf server_config.tar
)
