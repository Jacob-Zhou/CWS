exe="nohup python -u main.py"
$exe --device 4 --config config_zx.50.ini --path=exp/zx.50 --with_extra_dictionarys=False --is-train > results/log.create-dict.zx.50 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50 --with_extra_dictionarys=False --is-train > results/log.create-dict.sighan10.50 2>&1 

# no embedding & pre concat
$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=pre --is-train > results/log.create-dict.zx.50.aaai.ne_pre_concat 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=pre --is-train > results/log.create-dict.sighan10.50.aaai.ne_pre_concat 2>&1 

# no embedding & post concat
$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=post --is-train > results/log.create-dict.zx.50.aaai.ne_post_concat 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=False --dict-concat-type=post --is-train > results/log.create-dict.sighan10.50.aaai.ne_post_concat 2>&1 

# embedding & pre concat
$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-train > results/log.create-dict.zx.50.aaai.pre_concat 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-train > results/log.create-dict.sighan10.50.aaai.pre_concat 2>&1 

# embedding & post concat
$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-train > results/log.create-dict.zx.50.aaai.post_concat 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-train > results/log.create-dict.sighan10.50.aaai.post_concat 2>&1 

# ours embedding & pre concat
$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-train > results/log.create-dict.zx.50.cd.pre_concat 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=pre --is-train > results/log.create-dict.sighan10.50.cd.pre_concat 2>&1 

# ours embedding & post concat
$exe --device 4 --config config_zx.50.ini --path=exp/zx.50.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-train > results/log.create-dict.zx.50.cd.post_concat 2>&1 

$exe --device 4 --config config_sighan10.50.ini --path=exp/sighan10.50.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=True --with-dict-emb=True --dict-concat-type=post --is-train > results/log.create-dict.sighan10.50.cd.post_concat 2>&1 