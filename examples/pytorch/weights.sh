#!/bin/bash

max_jobs_per_gpu=2
available_gpus="0 1 2 3 4 5 6 7"

data_path="/net/iutd01/export/saito/dataset/onlyDiTau/"
event=50000
common_args="--data_path ${data_path} --event ${event} --conf config/config.yaml --is_gp_3dim False"

run_list=()
log_list=()
for i in {0..19}; do
    for weight in 0.0 0.001 0.01 0.1 0.5 0.9 0.99 0.999 1.0; do
        common_args2="-p weight --weight ${weight} -s ${i} ${common_args}"
        common_logs="weight${weight}.seed${i}"
        run_list+=("python main_simple.py ${common_args2}")
        log_list+=("log.SPOS_NAS.${common_logs}.dat")
    done
done

ijobs=0
igpu_last=-1
while true; do
    if [ ${ijobs} -eq ${#run_list[@]} ]; then
        break
    fi

    for igpu in ${available_gpus}; do
        if [ ${igpu} -eq ${igpu_last} ]; then
            continue
        fi
        njobs=$((`nvidia-smi -i ${igpu} --query-compute-apps=pid,process_name --format=csv | wc -l` - 1))
        if [ $njobs -lt ${max_jobs_per_gpu} ]; then
            job_cmnd=${run_list[ijobs]}
            logfile=${log_list[ijobs]}
            echo "ijobs: ${ijobs}, GPU:${igpu}, Run ${job_cmnd}"
            ${job_cmnd} --gpu_index ${igpu} >& ${logfile} &
            sleep 60
            ijobs=$(($ijobs + 1))
            igpu_last=${igpu}
            continue 2
        fi
    done

    sleep 120
done
