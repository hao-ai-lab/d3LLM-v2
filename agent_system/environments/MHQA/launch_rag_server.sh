# nohup bash launch_rag_server.sh > launch_rag_server_8000.log 2>&1 &

file_path=$HOME/efficient-verl-agent/data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=$file_path/models/e5-base-v2
port=8008

python ./agent_system/environments/MHQA/wiki_rag_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu \
    --host 127.0.0.1 \
    --port $port
