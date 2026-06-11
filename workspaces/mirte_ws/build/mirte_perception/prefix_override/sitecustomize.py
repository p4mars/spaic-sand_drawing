import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mirte/Spatial-AI-group-1/workspaces/mirte_ws/install/mirte_perception'
