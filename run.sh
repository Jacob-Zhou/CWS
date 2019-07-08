cudaN=4,5,6,7
exe="nohup python -u main.py --device $cudaN --config config.ini"
#$exe  --is-train --is-test > results/log.create-dict 2>&1 &
$exe --is-dictionary-exist --is-train > results/log.mtl.bert 2>&1 &

# num=`ls -d models-* | egrep -o '[0-9]+'`
# echo $num
#$exe  --is-test --model_eval_num $num > log.test-$num 2>&1

#CUDA_VISIBLE_DEVICES=$cudaN $exe  --is-test --model_eval_num $num #> log.test-$num 2>&1
