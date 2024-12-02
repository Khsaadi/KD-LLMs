base_path=${1-"/home/MiniLLM"}
port=2040


for data in dolly self_inst 
do
    # # Evaluate SelectiveMiniLLM
    for seed in 10 20 30 40 50
    do
        ckpt="gpt2-base"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done
done