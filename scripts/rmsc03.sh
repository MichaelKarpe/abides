#!/usr/bin/env bash

date=20190628
ticker="ABM"
experiment_name="POV_market_maker_param_sweep_quarter_day_value_arrivals_rate_7e-11_mom_wakeup_20s"

seeds="30, 31, 32, 33, 34, 35"
mm_povs="0.05"
mm_min_order_sizes="25"
mm_window_sizes="5"
mm_num_ticks="50"
mm_wake_up_freqs="'10S'"

num_runs=48
num_parallel_runs=8
python -u util/random_search.py -l ${seeds} -l ${mm_povs} -l ${mm_min_order_sizes} -l ${mm_window_sizes} -l ${mm_num_ticks} -l ${mm_wake_up_freqs} -n ${num_runs} > /tmp/vars.txt

while IFS= read -r line; do
    a=($(echo "$line" | tr ',' '\n'))
    seed="${a[0]}"
    pov="${a[1]}"
    min_order_size="${a[2]}"
    window_size="${a[3]}"
    num_ticks="${a[4]}"
    wake_up_freq="${a[5]}"
    baseline_log=${experiment_name}_${seed}_${date}_${pov}_${min_order_size}_${window_size}_${num_ticks}_${wake_up_freq}
    rm -f batch_output/${baseline_log}.err
    sem -j${num_parallel_runs} --line-buffer python -u abides.py -c market_maker_param_sweep \
                        -t ${ticker} \
                        -d $date \
                        -l ${baseline_log} \
                        -s ${seed} \
                        --wide-book \
                        --mm-pov ${pov} \
                        --mm-min-order-size ${min_order_size} \
                        --mm-window-size ${window_size} \
                        --mm-num-ticks ${num_ticks} \
                        --mm-wake-up-freq ${wake_up_freq} > batch_output/${baseline_log}.err 2>&1
#    stream="../../log/${baseline_log}/EXCHANGE_AGENT.bz2"
#    book="../../log/${baseline_log}/ORDERBOOK_${ticker}_FULL.bz2"
#    out_file="viz/${baseline_log}.png"
#    sem -j${num_parallel_runs} --line-buffer python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}
done < /tmp/vars.txt
sem --wait