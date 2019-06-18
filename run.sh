cudaN=2
exe="nohup python -u main.py --device cuda:$cudaN --config_file config.txt"
#$exe --is_dictionary_exist 0 --is_train 1 --is_test 0 > results/log.create-dict 2>&1 &
$exe --is_dictionary_exist 1 --is_train 1 --is_test 0 > results/log.mtl.bert 2>&1 &

# num=`ls -d models-* | egrep -o '[0-9]+'`
# echo $num
#$exe --is_train 0 --is_test 1 --model_eval_num $num > log.test-$num 2>&1

#CUDA_VISIBLE_DEVICES=$cudaN $exe --is_train 0 --is_test 1 --model_eval_num $num #> log.test-$num 2>&1
