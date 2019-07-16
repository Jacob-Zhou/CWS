exe="nohup python -u main.py"

# $exe --device 4 --config config_zx.ini --path=exp/zx --is-dictionary-exist --is-train > results/log.zx 2>&1 
# $exe --device 4 --config config_zx_test.ini --path=exp/zx --is-test > results/log.test.zx 2>&1 
# $exe --device 4 --config config_zx.bert.ini --path=exp/zx.bert --is-train > results/log.create-dict.zx.bert 2>&1 
# $exe --device 4 --config config_zx.bert.ini --path=exp/zx.bert --is-dictionary-exist --is-train > results/log.zx.bert 2>&1 
$exe --device 4 --config config_zx_test.bert.ini --path=exp/zx.bert --is-test > results/log.test.zx.bert 2>&1 