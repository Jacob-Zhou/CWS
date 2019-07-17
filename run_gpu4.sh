exe="nohup python -u main.py"

# $exe --device 4 --config config_zx.ini --path=exp/zx.cd --is-dictionary-exist --is-train > results/log.cd.zx 2>&1 
# $exe --device 4 --config config_zx_test.ini --path=exp/zx.cd --is-test > results/log.test.cd.zx 2>&1 
# $exe --device 4 --config config_zx.bert.ini --path=exp/zx.cd.bert --is-train > results/log.create-dict.zx.bert 2>&1 
# $exe --device 4 --config config_zx.bert.ini --path=exp/zx.cd.bert --is-dictionary-exist --is-train > results/log.zx.cd.bert.dic_cat 2>&1 
$exe --device 4 --config config_zx_test.bert.ini --path=exp/zx.cd.bert --is-test > results/log.test.zx.cd.bert.lr_4 2>&1 &