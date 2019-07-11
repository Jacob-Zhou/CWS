cudaN=5
exe="nohup python -u main.py --device $cudaN --config config.ini --path=exp/ctb9-mtl-bert"
# $exe  --is-train > results/log.create-dict 2>&1 &
# $exe --is-dictionary-exist --is-train > results/log.mtl.bert 2>&1 &

$exe  --is-test > log.test 2>&1 &
