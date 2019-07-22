exe="python -u main.py"

# echo "embedding & pre concat"

# $exe --device 6 --config config_zx.50.ini --path=exp/zx.50.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.50.aaai.pre_concat 2>&1 

# $exe --device 6 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.50.aaai.pre_concat 2>&1 

# echo "embedding & post concat"

# $exe --device 6 --config config_zx.50.ini --path=exp/zx.50.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.50.aaai.post_concat 2>&1 

# $exe --device 6 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.50.aaai.post_concat 2>&1 

echo "ours embedding & pre concat"

$exe --device 6 --config config_zx.50.ini --path=exp/zx.50.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.50.cd.pre_concat 2>&1 

$exe --device 6 --config config_sighan10.50.ini --path=exp/sighan10.50.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.50.cd.pre_concat 2>&1 

echo "ours embedding & post concat"

$exe --device 6 --config config_zx.50.ini --path=exp/zx.50.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.50.cd.post_concat 2>&1 

$exe --device 6 --config config_sighan10.50.ini --path=exp/sighan10.50.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.50.cd.post_concat 2>&1 