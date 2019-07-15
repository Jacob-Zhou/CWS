exe="nohup python -u main.py"
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10 --is-train > results/log.create-dict.sighan10 2>&1 &
# $exe --device 5 --config config_zx.ini --path=exp/zx --is-train > results/log.create-dict.zx 2>&1 &

$exe --device 5 --config config_sighan10.ini --path=exp/sighan10 --is-dictionary-exist --is-train > results/log.sighan10 2>&1 &
# $exe --device 5 --config config_zx.ini --path=exp/zx --is-dictionary-exist --is-train > results/log.zx 2>&1 &
# $exe --device 5 --config config_zx_test.ini --path=exp/zx --is-test > results/log.test.zx 2>&1 &