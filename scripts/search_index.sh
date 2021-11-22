python -m pyserini.search --topics podcasts_2020_topics_train_description.tsv  --output run.desc.train.txt --index indexes/sample_collection_jsonl --bm25
python -m pyserini.search --topics podcasts_2020_topics_train_query.tsv  --output run.query.train.txt --index indexes/sample_collection_jsonl --bm25
python -m pyserini.search --topics podcasts_2020_topics_test_description.tsv  --output run.desc.test.txt --index indexes/sample_collection_jsonl --bm25
python -m pyserini.search --topics podcasts_2020_topics_test_query.tsv  --output run.query.test.txt --index indexes/sample_collection_jsonl --bm25
