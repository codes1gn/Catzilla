#!/bin/bash

# Define an array
# N C H W K R S P Q U V iteration warmup
declare -a params=(
  "8 32 224 224 128 3 3 1 1 1 1 10000 10"
#  "8 64 56 56 64 3 3 1 1 1 1 1000 10"
#  "8 128 28 28 128 3 3 1 1 2 2 1000 10"
#  "8 256 14 14 256 3 3 1 1 1 1 1000 10"
  #  "8 32 224 224 128 3 3 1 1 1 1"
)

#build_dir='cmake-build-release/'
build_dir='Build/Release/'

run_implgemm=true
run_pytorch=true
run_cudnn_frontend=true
run_cutlass=true

# Print all elements of the array
for i in "${params[@]}"
do
  # Split element into an array
  arr=($i)

  # Assign each value to a variable
  N=${arr[0]}
  C=${arr[1]}
  H=${arr[2]}
  W=${arr[3]}
  K=${arr[4]}
  R=${arr[5]}
  S=${arr[6]}
  P=${arr[7]}
  Q=${arr[8]}
  U=${arr[9]}
  V=${arr[10]}
  iteration=${arr[11]}
  warmup=${arr[12]}

  # Print each variable
  echo "N: $N, C: $C, H: $H, W: $W, K: $K, R: $R, S: $S, P: $P, Q: $Q, U: $U, V: $V"

  # 自己写implicitGEMM
  if $run_implgemm ; then
    echo "implgemm"
    echo "$build_dir/src/SpConv/implgemm -N $N -C $C -H $H -W $W -K $K -R $R -S $S -P $P -Q $Q -U $U -V $V -I $iteration -Warmup $warmup"
    $build_dir/src/SpConv/implgemm -N $N -C $C -H $H -W $W -K $K -R $R -S $S -P $P -Q $Q -U $U -V $V -I $iteration -Warmup $warmup
  fi

  # pytorch
  if $run_pytorch ; then
    echo "pytorch"
    echo "python src/baselines/pytorch/main.py -N $N -C $C -H $H -W $W -K $K -R $R -S $S -P $P -Q $Q -U $U -V $V -I $iteration --warmup $warmup"
    python src/baselines/pytorch/main.py -N $N -C $C -H $H -W $W -K $K -R $R -S $S -P $P -Q $Q -U $U -V $V -I $iteration --warmup $warmup
  fi

  # cudnn
  if $run_cudnn_frontend ; then
    echo "cudnn_frontend"
    echo "$build_dir/src/baselines/cudnn_frontend/conv_cudnn_frontend -N $N -C $C -H $H -W $W -K $K -R $R -S $S -P $P -Q $Q -U $U -V $V"
    $build_dir/src/baselines/cudnn_frontend/conv_cudnn_frontend -N $N -C $C -H $H -W $W -K $K -R $R -S $S -P $P -Q $Q -U $U -V $V
  fi

  # cutlass
  if $run_cutlass ; then
    echo "cutlass"
    echo "$build_dir/src/baselines/cutlass/conv_cutlass_float --n=$N --c=$C --h=$H --w=$W --k=$K --r=$R --s=$S --iterations=$iteration"
    $build_dir/src/baselines/cutlass/conv_cutlass_float --n=$N --c=$C --h=$H --w=$W --k=$K --r=$R --s=$S --iterations=$iteration
  fi
done