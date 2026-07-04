"""
fm_dad — Full-Mode DRL-Based Adaptive Defense package.

Four independent DRL agents for vehicular network security:
    SP  = Selective Packet Dropping
    ALS = Asymmetric Link Spoofing
    IGH = Interleaved Grey Hole
    FS  = Flow Stretching

Each agent is a separate DQNAgent instance with its own networks,
replay buffer, and optimizer (R1: no weight sharing between agents).
"""
