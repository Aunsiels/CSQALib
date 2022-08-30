all: obqa/train.jsonl csqa/train_rand_split.jsonl cpnet-rels.tsv ascentpp.csv cpnet_en.csv quasimodo_positive_top.tsv

obqa/train.jsonl:
	wget https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip
	unzip OpenBookQA-V1-Sep2018.zip
	mv OpenBookQA-V1-Sep2018/Data/Main obqa
	rm -r OpenBookQA-V1-Sep2018 OpenBookQA-V1-Sep2018.zip

csqa/train_rand_split.jsonl:
	wget -nc -P csqa https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
	wget -nc -P csqa https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
	wget -nc -P csqa https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl

cpnet-rels.tsv:
	echo you should copy paste cpnet-rels.tsv

ascentpp.csv:
	wget https://www.mpi-inf.mpg.de/fileadmin/inf/d5/research/ascentpp/ascentpp.csv.tar.gz
	tar -xvzf ascentpp.csv.tar.gz
	rm ascentpp.csv.tar.gz

cpnet_en.csv:
	# gsutil cp gs://aotoml-tmp/csqa/cpnet_en.csv cpnet_en.csv

	wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
	gunzip conceptnet-assertions-5.7.0.csv.gz
	cat conceptnet-assertions-5.7.0.csv | awk '$$3 ~ "^/c/en/" && $$4 ~ "^/c/en/"' > cpnet_en.csv

quasimodo_positive_top.tsv:
	wget -O quasimodo_positive_top.tsv https://nextcloud.mpi-klsb.mpg.de/index.php/s/Sioq6rKP8LmjMDQ/download?path=%2FLatest%2fquasimodo_positive_top.tsv
