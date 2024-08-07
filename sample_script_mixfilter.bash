device=0
dataset=DomainNet
steps=15000
ffcv=False

lr=5e-5
wd=1e-4
rd=0.5
drate=0.6
grate=0.1

for seed in 0 1 2
do
    for test_envs in 0 1 2 3 4 5
    do 
        name=R50_env${test_envs}_s${seed}_mixv2_lr_${lr}_wd_${wd}_gp_${grate}
        files=($(ls train_output/$dataset | grep $name))
        if [ ${#files[@]} -gt 0 ]; then
            echo "Skipping $name"
            continue
        fi

        python train_all.py $name --test_envs $test_envs \
        --device $device --lr $lr --weight_decay $wd --algorithm ERM \
        --rx True --steps $steps --resnet_dropout $rd --group_dropout $grate \
        --deterministic --dataset $dataset --trial_seed $seed --drop_mode filter --drop_activation False --drop_noise 0 --ffcv $ffcv
    done
done

for seed in 0 2
do
    for test_envs in 3 4 5
    do
        name=R50_env${test_envs}_s${seed}_mixv2+ma_lr_${lr}_wd_${wd}_gp_${grate}
        files=($(ls train_output/$dataset | grep $name))
        if [ ${#files[@]} -gt 0 ]; then
            echo "Skipping $name"
            continue
        fi

        python train_all.py $name --test_envs $test_envs \
        --device $device --lr $lr --weight_decay $wd --algorithm MA \
        --rx True --steps $steps --resnet_dropout $rd --group_dropout $grate \
        --deterministic --dataset $dataset --trial_seed $seed --drop_mode filter --drop_activation False --drop_noise 0 --ffcv $ffcv
    done
done