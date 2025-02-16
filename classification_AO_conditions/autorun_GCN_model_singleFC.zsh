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

# Split each pane vertically into 2 smaller panes (4 panes in total)
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 2
tmux split-window -v

# In the first pane, run 'watch -n 1 nvidia-smi'
tmux select-pane -t 0
tmux send-keys "watch -n 1 nvidia-smi" C-m

# In the second pane, run 'python3 GCN_model_singleFC.py --FC_name COH' then 'python3 GCN_model_singleFC.py --FC_name iCOH'
tmux select-pane -t 1
tmux send-keys "python3 GCN_model_singleFC.py --FC_name COH && python3 GCN_model_singleFC.py --FC_name iCOH" C-m

# In the third pane, run 'python3 GCN_model_singleFC.py --FC_name PLV' then 'python3 GCN_model_singleFC.py --FC_name PDC'
tmux select-pane -t 2
tmux send-keys "python3 GCN_model_singleFC.py --FC_name PLV && python3 GCN_model_singleFC.py --FC_name PDC" C-m

# In the fourth pane, run 'python3 GCN_model_singleFC.py --FC_name iCOH'
tmux select-pane -t 3
tmux send-keys "python3 GCN_model_singleFC.py --FC_name iCOH" C-m

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME