exe="nohup python -u main.py"

# $exe --device 5 --config config_sighan10.ini --path=exp/sighan10 --is-dictionary-exist --is-train > results/log.sighan10 2>&1
$exe --device 3 --config config_sighan10.bert.ini --path=exp/sighan10.bert --is-train > results/log.sighan10.bert 2>&1
$exe --device 3 --config config_sighan10.bert.ini --path=exp/sighan10.bert --is-dictionary-exist --is-train > results/log.sighan10.bert 2>&1