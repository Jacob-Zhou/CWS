exe="python -u main.py"

$exe --device 1 --config config_zx.ini --path=exp/zx --with_extra_dictionarys=False --is-dictionary-exist --is-train > results/log.zx 2>&1 

$exe --device 1 --config config_sighan10.ini --path=exp/sighan10 --with_extra_dictionarys=False --is-dictionary-exist --is-train > results/log.sighan10 2>&1 

echo "no embedding & pre concat"

$exe --device 1 --config config_zx.ini --path=exp/zx.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.aaai.ne_pre_concat 2>&1 

$exe --device 1 --config config_sighan10.ini --path=exp/sighan10.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.aaai.ne_pre_concat 2>&1 

echo "no embedding & post concat"

$exe --device 1 --config config_zx.ini --path=exp/zx.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.aaai.ne_post_concat 2>&1 

$exe --device 1 --config config_sighan10.ini --path=exp/sighan10.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.aaai.ne_post_concat 2>&1 

# echo "embedding & pre concat"

# $exe --device 1 --config config_zx.ini --path=exp/zx.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.aaai.pre_concat 2>&1 

# $exe --device 1 --config config_sighan10.ini --path=exp/sighan10.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.aaai.pre_concat 2>&1 

# echo "embedding & post concat"

# $exe --device 1 --config config_zx.ini --path=exp/zx.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.aaai.post_concat 2>&1 

# $exe --device 1 --config config_sighan10.ini --path=exp/sighan10.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.aaai.post_concat 2>&1 

# echo "ours embedding & pre concat"

# $exe --device 1 --config config_zx.ini --path=exp/zx.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.cd.pre_concat 2>&1 

# $exe --device 1 --config config_sighan10.ini --path=exp/sighan10.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.cd.pre_concat 2>&1 

# echo "ours embedding & post concat"

# $exe --device 1 --config config_zx.ini --path=exp/zx.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.cd.post_concat 2>&1 

# $exe --device 1 --config config_sighan10.ini --path=exp/sighan10.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.cd.post_concat 2>&1 