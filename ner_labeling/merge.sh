rm -rf merge
mkdir merge

function merge
{
    param=$1
    cp $param merge
    cat cons/$param  >> merge/$param
    sort -u merge/$param > tmp
    mv tmp merge/$param
    echo $param
    wc -l $param cons/$param merge/$param
}

merge DISEASE
merge DRUG
merge GENE
merge LOCATION
merge ORGANIZATION
merge PERSON
merge PROTEIN
merge SOCIAL_CIRCUMSTANCES
merge SPECIES
