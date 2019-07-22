exe="nohup python -u main.py"

# create dict

$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.cd --is-train

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.cd --is-train

$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.aaai.embed --dict-feature-type=aaai --is-train

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.embed --dict-feature-type=aaai --is-train

# train

$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.cd --is-dictionary-exist --is-train > results/log.zx.50.cd 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.cd --is-dictionary-exist --is-train > results/log.sighan10.50.cd 2>&1 

$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.aaai.embed --dict-feature-type=aaai --is-dictionary-exist --is-train > results/log.zx.50.aaai.embed 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.embed --dict-feature-type=aaai --is-dictionary-exist --is-train > results/log.sighan10.50.aaai.embed 2>&1 