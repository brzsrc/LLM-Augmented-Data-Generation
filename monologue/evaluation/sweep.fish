#!/usr/bin/env fish
# Grid sweep over (gamma, cql_alpha) for policy_utility_kfold.py
# Run from monologue/evaluation/:
#     ./sweep.fish
# Or detached:
#     nohup fish sweep.fish > outputs/sweep/master.log 2>&1 &

cd (dirname (status filename))
mkdir -p outputs/logs

for g in 0.9 0.95 0.99
    for a in 0.5 1.0 2.0 5.0
        set tag "g{$g}_a{$a}"
        echo ">>> [$tag] start at "(date)
        python policy_utility_kfold.py \
            --gamma $g --cql_alpha $a \
            --out_dir ./outputs/$tag \
            > outputs/logs/$tag.log 2>&1
        echo ">>> [$tag] done at "(date)
    end
end

echo ">>> sweep complete"
