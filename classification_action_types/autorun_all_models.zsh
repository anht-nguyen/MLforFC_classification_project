#!/bin/zsh

# Check if a session name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <session_name>"
  exit 1
fi

SESSION_NAME=$1

# Start a new tmux session with the provided session name
tmux new-session -d -s $SESSION_NAME

# Split the window horizontally into 2 panes
tmux split-window -h

# In the first pane, run 'watch -n 1 nvidia-smi' and then run MLP and GCN models
tmux select-pane -t 0
tmux send-keys "watch -n 1 nvidia-smi" C-m
tmux send-keys "sh ./autorun_MLP_model_singleFC.zsh runMLP && sh ./autorun_GCN_model_singleFC.zsh runGCN" C-m

# In the second pane, run CNN models and then run ML models training and validating script
tmux select-pane -t 1
tmux send-keys "sh ./autorun_CNN_model_singleFC.zsh runCNN && python ML_models_training_validating.py" C-m

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME