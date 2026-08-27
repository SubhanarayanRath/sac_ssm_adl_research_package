import numpy as np

def main():
    print("--- NPZ Inspection Script ---")
    filepath = "data/processed/aruba_tar_windows.npz"
    
    try:
        data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Could not find {filepath}. Please check the path.")
        return

    print("\n1. NPZ Arrays (Keys):")
    print(data.files)

    print("\n2. y_activities Raw Data (first 3 windows):")
    print(data["y_activities"][:3])

    print("\n3. y_activities Data Type (first element):")
    print(type(data["y_activities"][0, 0]))

    # Replicate the exact encode_strings logic from your loader
    def encode_strings(array):
        unique_vals = np.unique(array)
        vocab_map = {val: i for i, val in enumerate(unique_vals)}
        mapped = np.vectorize(vocab_map.get)(array)
        return mapped.astype(np.int32), len(unique_vals)

    y_activity_seq = data["y_activities"]
    y_activity_last, num_classes = encode_strings(y_activity_seq[:, -1])

    print("\n4. Encoded y_activity_last Unique Values:")
    print(np.unique(y_activity_last))

    print("\n5. Encoded y_activity_last (first 20):")
    print(y_activity_last[:20])

if __name__ == "__main__":
    main()
