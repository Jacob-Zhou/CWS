cudaN=6,7
exe="nohup python -u main.py --device $cudaN --config config.ini --path=exp/ctb9-mtl-bert"
# $exe  --is-train > results/log.create-dict 2>&1 &
$exe --is-dictionary-exist --is-train > results/log.mtl.bert 2>&1 &

# num=`ls -d models-* | egrep -o '[0-9]+'`
# echo $num
# $exe  --is-test --model-eval-num=$num > log.test-$num 2>&1
