# config.py

CONFIG = {
    "input_size": (256, 256),
    "channels": 3,
    "batch_size": 4,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "num_memory_items": 512,
    "memory_item_size": 512,
    "memory_update_rate": 0.1,
    "bml_weight": 0.1,
    "num_heads": 4,
    "device": "cuda"  # or "cpu"
}
