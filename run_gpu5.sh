exe="nohup python -u main.py"

# $exe --device 5 --config config_sighan10.ini --path=exp/sighan10.cd --is-dictionary-exist --is-train > results/log.cd.sighan10 2>&1
$exe --device 3 --config config_sighan10.bert.ini --path=exp/sighan10.cd.bert --is-train > results/log.sighan10.cd.bert 2>&1
$exe --device 3 --config config_sighan10.bert.ini --path=exp/sighan10.cd.bert --is-dictionary-exist --is-train > results/log.sighan10.cd.bert 2>&1