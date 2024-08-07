device=1
dataset=DomainNet
steps=15000
ffcv=False

for seed in 1
do
    for test_envs in 0 1 2 3 4 5
    do
        files=($(ls train_output/$dataset/* -1 -d | grep s${seed} | grep env${test_envs} | grep erm_))

        name=R50_env${test_envs}_s${seed}_erm_wa_1

        python train_all.py $name --test_envs $test_envs \
        --device $device --lr 0 --weight_decay 0 --algorithm DiWA \
        --rx True --steps 1 --resnet_dropout 0 --group_dropout 0 --average_weights True --ens_size 20 \
        --deterministic --dataset $dataset --trial_seed $seed --drop_mode filter --drop_activation False --drop_noise 0 --ffcv $ffcv --model_test ${files[@]}

    done
done
