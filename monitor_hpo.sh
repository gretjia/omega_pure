#!/bin/bash
JOB_ID=$1
PROJECT="gen-lang-client-0250995579"
REGION="us-central1"
LOG_FILE="/home/zephryj/projects/omega_pure/hpo_monitor.log"

echo "==========================================================" > $LOG_FILE
echo "🚀 OMEGA PROTOCOL: HPO JOB MONITOR" >> $LOG_FILE
echo "Job ID: $JOB_ID" >> $LOG_FILE
echo "ETA: ~1.0 to 1.5 Hours | Polling Interval: 2 minutes" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE

while true; do
    INFO=$(gcloud ai hp-tuning-jobs describe $JOB_ID --project=$PROJECT --region=$REGION --format="json" 2>/dev/null)
    STATE=$(echo "$INFO" | jq -r '.state')
    
    # 统计 trial 状态
    COMPLETED=$(echo "$INFO" | jq -r '.trials[]?.state' | grep -c "SUCCEEDED")
    FAILED=$(echo "$INFO" | jq -r '.trials[]?.state' | grep -c "FAILED\|INFEASIBLE")
    ACTIVE=$(echo "$INFO" | jq -r '.trials[]?.state' | grep -c "ACTIVE")
    TOTAL=$((COMPLETED + FAILED + ACTIVE))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 状态: $STATE | 正在探索: $ACTIVE | 已寻优: $COMPLETED | 失败/不可行: $FAILED | 总计: $TOTAL/15" >> $LOG_FILE
    
    if [[ "$STATE" == "JOB_STATE_SUCCEEDED" || "$STATE" == "JOB_STATE_FAILED" || "$STATE" == "JOB_STATE_CANCELLED" ]]; then
        echo "==========================================================" >> $LOG_FILE
        echo "🏁 寻优任务结束，最终状态: $STATE" >> $LOG_FILE
        
        # 提取最优参数 (如果有)
        if [[ "$COMPLETED" -gt 0 ]]; then
             echo "最优物理常数 (Best Parameters):" >> $LOG_FILE
             BEST_PARAMS=$(echo "$INFO" | jq -r '[.trials[]? | select(.state=="SUCCEEDED")] | min_by(.finalMeasurement.metrics[0].value) | {trial_id: .id, val_fvu: .finalMeasurement.metrics[0].value, seq_len: (.parameters[] | select(.parameterId=="seq_len").value), stride: (.parameters[] | select(.parameterId=="stride").value)}')
             echo "$BEST_PARAMS" >> $LOG_FILE
        fi
        
        break
    fi
    
    # 轮询周期 120 秒
    sleep 120
done
