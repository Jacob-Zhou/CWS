cudaN=5
exe="nohup python -u main.py"
# $exe --device 2 --config config_as.ini --path=exp/as.bert --is-train > results/log.create-dict.as.bert 2>&1 &
# $exe --device 1 --config config_cityu.ini --path=exp/cityu.bert --is-train > results/log.create-dict.cityu.bert 2>&1 &

$exe --device 2 --config config_as.ini --path=exp/as.bert --is-dictionary-exist --is-train > results/log.as.bert 2>&1 &
$exe --device 1 --config config_cityu.ini --path=exp/cityu.bert --is-dictionary-exist --is-train > results/log.cityu.bert 2>&1 &
