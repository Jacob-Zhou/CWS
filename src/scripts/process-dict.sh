
#python collect_word_and_freq.py --inf Giga.CTB5.len-less-than-150.hwc.coarse-and-fine --out_dict_file Giga-coarse-fine.dict.txt 2> Giga-coarse-fine.words.txt
paste Giga-coarse-fine.dict.txt subword-nmt-master/Giga-bpe-out > Giga-coarse-fine.dict-with-bpe.txt

