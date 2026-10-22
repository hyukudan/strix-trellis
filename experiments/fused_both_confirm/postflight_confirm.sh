date -Is
sha256sum -c build/frozen.sha256 > build/postflight-frozen.txt
sha256sum -c build/fixtures.sha256 > build/postflight-fixtures.txt
if [[ $(hostname) == node2 ]];then
    sha256sum -c .old-baseline-node2.sha256 > build/postflight-old-preserved.txt
else
    sha256sum -c build/old-preservation.sha256 > build/postflight-old-preserved.txt
fi
bash experiments/fused_both_confirm/observe_confirm.sh
