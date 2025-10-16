from json_handle import concat_concept_matrices
from manifold import compute_block_means, two_NN, simple_two_NN
import numpy as np

if __name__ == "__main__":
    json_files = ["/test_210m20250928-113816ICA_concepts_retrieval/test_210m20250928-113816.json"]
    layers = ["model.layers.1.mlp", "model.layers.2.mlp", "model.layers.3.mlp", "model.layers.4.mlp", "model.layers.5.mlp", "model.layers.6.mlp", "model.layers.7.mlp", "model.layers.8.mlp", "model.layers.9.mlp", "model.layers.10.mlp", "model.layers.11.mlp"]
    step_keys_first = ["step"+str(i*10000) for i in range(1, 25)]
    step_keys_second = ["step"+str(i*10000) for i in range(25, 49)]
    for layer in layers:
        m = concat_concept_matrices(
                file_paths=json_files,
                step_keys=step_keys_first,
                layer_keys=[layer],
                axis=0,
                allow_missing=True,
                return_numpy=True,
            )
        print(f"\nLayer: {layer}")
        print(f"Matrix shape: {m.shape}")

        # Compute block means (mean of every 24 rows)
        m_prime = compute_block_means(m, block_size=m.shape[0]//100)
        print(f"Block means matrix shape: {m_prime.shape}")
        
        dim_prime = two_NN(m_prime, k=2)
        print(f"Estimated intrinsic dimension for {layer} (first regime, block means): {dim_prime:.2f}")
        
        m = concat_concept_matrices(
                file_paths=json_files,
                step_keys=step_keys_second,
                layer_keys=[layer],
                axis=0,
                allow_missing=True,
                return_numpy=True,
            )
        print(f"Matrix shape: {m.shape}")

        # Compute block means (mean of every 24 rows)
        m_prime = compute_block_means(m, block_size=m.shape[0]//100)
        print(f"Block means matrix shape: {m_prime.shape}")
        
        dim_prime = two_NN(m_prime, k=2)
        print(f"Estimated intrinsic dimension for {layer} (second regime, block means): {dim_prime:.2f}\n")