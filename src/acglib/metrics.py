import numpy as np

def dsc(output, target, background_label=0):
    """Dice Similarity Coefficient. Output and target must contain integer labels."""
    assert output.shape == target.shape, '{output.shape} != {target.shape}'
   
    output_foreground = np.not_equal(output, background_label)
    target_foreground = np.not_equal(target, background_label)

    intersection = np.sum(
        np.logical_and(
            np.equal(output, target), 
            np.logical_or(output_foreground, target_foreground)))
            
    denominator = np.sum(output_foreground) + np.sum(target_foreground)

    return 2.0 * intersection / denominator if denominator > 0.0 else 0.0 * intersection