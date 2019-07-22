exe="nohup python -u main.py"

# $exe --device 4 --config config_zx.ini --path=exp/zx --with_extra_dictionarys=false --is-train > results/log.create-dict.zx 2>&1 
$exe --device 4 --config config_zx.ini --path=exp/zx --with_extra_dictionarys=false --is-dictionary-exist --is-train > results/log.zx 2>&1 

$exe --device 4 --config config_sighan10.ini --path=exp/sighan10 --with_extra_dictionarys=false --is-train > results/log.create-dict.sighan10 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10 --with_extra_dictionarys=false --is-dictionary-exist --is-train > results/log.sighan10 2>&1 

# no embedding & pre concat
$exe --device 4 --config config_zx.ini --path=exp/zx.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=pre --is-train > results/log.create-dict.zx.aaai.ne_pre_concat 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.aaai.ne_pre_concat 2>&1 

$exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=pre --is-train > results/log.create-dict.sighan10.aaai.ne_pre_concat 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.ne_pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.aaai.ne_pre_concat 2>&1 

# no embedding & post concat
$exe --device 4 --config config_zx.ini --path=exp/zx.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=post --is-train > results/log.create-dict.zx.aaai.ne_post_concat 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.aaai.ne_post_concat 2>&1 

$exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=post --is-train > results/log.create-dict.sighan10.aaai.ne_post_concat 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.ne_post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=false --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.aaai.ne_post_concat 2>&1 

# embedding & pre concat
$exe --device 4 --config config_zx.ini --path=exp/zx.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-train > results/log.create-dict.zx.aaai.pre_concat 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.aaai.pre_concat 2>&1 

$exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-train > results/log.create-dict.sighan10.aaai.pre_concat 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.pre_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.aaai.pre_concat 2>&1 

# embedding & post concat
$exe --device 4 --config config_zx.ini --path=exp/zx.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-train > results/log.create-dict.zx.aaai.post_concat 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.aaai.post_concat 2>&1 

$exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-train > results/log.create-dict.sighan10.aaai.post_concat 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.post_concat --dict-feature-type=aaai --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.aaai.post_concat 2>&1 

# ours embedding & pre concat
$exe --device 4 --config config_zx.ini --path=exp/zx.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-train > results/log.create-dict.zx.cd.pre_concat 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.zx.cd.pre_concat 2>&1 

$exe --device 4 --config config_sighan10.ini --path=exp/sighan10.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-train > results/log.create-dict.sighan10.cd.pre_concat 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.cd.pre_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=pre --is-dictionary-exist --is-train > results/log.sighan10.cd.pre_concat 2>&1 

# ours embedding & post concat
$exe --device 4 --config config_zx.ini --path=exp/zx.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-train > results/log.create-dict.zx.cd.post_concat 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-dictionary-exist --is-train > results/log.zx.cd.post_concat 2>&1 

$exe --device 4 --config config_sighan10.ini --path=exp/sighan10.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-train > results/log.create-dict.sighan10.cd.post_concat 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.cd.post_concat --dict-feature-type=ours --with_extra_dictionarys=true --with-dict-emb=true --dict-concat-type=post --is-dictionary-exist --is-train > results/log.sighan10.cd.post_concat 2>&1 

# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai --dict-feature-type=aaai --is-train > results/log.create-dict.zx.aaai 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai --dict-feature-type=aaai --is-dictionary-exist --is-train > results/log.zx.aaai 2>&1 

# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai --dict-feature-type=aaai --is-train > results/log.create-dict.sighan10.aaai 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai --dict-feature-type=aaai --is-dictionary-exist --is-train > results/log.sighan10.aaai 2>&1 

# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai.embed --dict-feature-type=aaai --is-train > results/log.create-dict.zx.aaai.embed 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.aaai.embed --dict-feature-type=aaai --is-dictionary-exist --is-train > results/log.zx.aaai.embed 2>&1 

# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.embed --dict-feature-type=aaai --is-train > results/log.create-dict.sighan10.aaai.embed 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.aaai.embed --dict-feature-type=aaai --is-dictionary-exist --is-train > results/log.sighan10.aaai.embed 2>&1 

# $exe --device 4 --config config_zx.ini --path=exp/zx.cd.e5 --is-train > results/log.create-dict.zx.cd 2>&1 
# $exe --device 4 --config config_zx.ini --path=exp/zx.cd.e5 --is-dictionary-exist --is-train > results/log.zx.cd 2>&1 

# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.cd.e5 --is-train > results/log.create-dict.sighan10.cd 2>&1 
# $exe --device 4 --config config_sighan10.ini --path=exp/sighan10.cd.e5 --is-dictionary-exist --is-train > results/log.sighan10.cd 2>&1 

# $exe --device 4 --config config_zx_test.ini --path=exp/zx.cd --is-test > results/log.test.cd.zx 2>&1 
# $exe --device 4 --config config_zx.bert.ini --path=exp/zx.cd.bert --is-train > results/log.create-dict.zx.bert 2>&1 

# $exe --device -1 --config config_zx.bert.ms.ini --path=exp/zx.ms.mlt.bert --is-train > results/log.create-dict.zx.ms.mlt.bert 2>&1 
# $exe --device 4 --config config_zx.bert.ms.ini --path=exp/zx.ms.mlt.bert --is-dictionary-exist --is-train > results/log.zx.ms.mlt.bert 2>&1 

# $exe --device 4 --config config_zx.ms.ini --path=exp/zx.ms.mlt --is-train > results/log.zx.ms.mlt 2>&1 
# $exe --device 4 --config config_zx.ms.ini --path=exp/zx.ms.mlt --is-dictionary-exist --is-train > results/log.zx.ms.mlt 2>&1 
# $exe --device 4 --config config_zx.ms.ini --path=exp/zx.ms.mlt --is-test > results/log.test.zx.ms.mlt 2>&1 

# $exe --device 4 --config=config_zx.bert.mlt.ini --path=exp/zx.bert.mlt --is-train > results/log.create-dict.zx.bert.mlt 2>&1 
# $exe --device 4 --config=config_zx.bert.mlt.ini --path=exp/zx.bert.mlt --is-dictionary-exist --is-train > results/log.zx.bert.mlt 2>&1 

# $exe --device 4 --config=config_zx.mlt.ini --path=exp/zx.mlt --is-train > results/log.create-dict.zx.mlt 2>&1 
# $exe --device 4 --config=config_zx.mlt.ini --path=exp/zx.mlt --is-dictionary-exist --is-train > results/log.zx.mlt 2>&1 
# $exe --device 4 --config config_zx.ms.ini --path=exp/zx.ms.mlt --is-test > results/log.test.zx.ms.mlt 2>&1 

# $exe --device 4 --config=config_sighan10.bert.ms.ini --path=exp/sighan10.ms --is-train > results/log.create-dict.sighan10.ms 2>&1 
# $exe --device 4 --config=config_sighan10.bert.ms.ini --path=exp/sighan10.ms --is-dictionary-exist --is-train > results/log.sighan10.ms 2>&1 


# $exe --device 1 --config config_zx.bert.ms.ini --path=exp/zx.ms.bert --is-test > results/log.test.zx.ms.bert.ctb5_bpe 2>&1 
# $exe --device 4 --config config_zx_test.bert.ini --path=exp/zx.cd.bert --is-test > results/log.test.zx.cd.bert.lr_4 2>&1 &