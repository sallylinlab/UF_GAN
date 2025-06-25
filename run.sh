#! /bin/bash

### create folders
listFolder=("result" "saved_model")

for t in ${listFolder[@]}; do
    echo "Create folder $t if not exists"
    mkdir -p $t
done

### run experiments

listShots=(20 1)

listDataset=(
    "mura"
)    

echo "start training process for mura april resized mid dataset with resnext50"

for ns in ${listShots[@]}; do
    for t in ${listDataset[@]}; do
        version=0001
        echo "Start Program $t of version $version with shot $ns"
        res_dir="result/$t""_$version-$ns/"
        saved_model_dir="saved_model/$t""_v$version-$ns/"
        # echo $res_dir
        mkdir -p $res_dir
        mkdir -p $saved_model_dir

        # run programming
        python3 main.py -dn $t -s $ns -m true -rd $res_dir -ted "test_data" -trd "train_data" -eld "eval_data" -smd $saved_model_dir -rtd "data" -sz 128

        sleep 30
        echo "Oops! I fell asleep for a couple seconds!"
    done
done

