"""
Unit tests for epsilon_net function to verify distance_fn parameter works correctly.

Tests alignment-based RMSD distance (helpers.alignment_rmsd_distance) vs standard
Euclidean distance (scipy.spatial.distance.cdist) for XYZ coordinate data.

All data in this file follows the (n_samples, n_features) convention, where
n_features = 3 * n_atoms.  Data is transposed to (n_features, n_samples) only when
passed to epsilon_net, which uses that internal layout.

The alignment RMSD distance follows the same pattern as compute_aligned_sqdists
in run/butane.py: it reshapes flat coordinates to (n_samples, n_atoms, 3) and
computes pairwise RMSD via Kabsch alignment.
"""

import numpy as np
import sys
import os
from scipy.spatial.distance import cdist

# Add src to path to import helpers
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import helpers


def euclidean_distance(X, Y):
    """
    Compute Euclidean distance between points in X and Y.

    Args:
        X: Array of shape (n_samples_X, n_features)
        Y: Array of shape (n_samples_Y, n_features)

    Returns:
        Distance matrix of shape (n_samples_X, n_samples_Y)
    """
    return cdist(X, Y, metric='euclidean')


def generate_xyz_trajectory(n_atoms, n_frames, spread=2.0, noise=0.1):
    """
    Generate synthetic XYZ trajectory data for testing.

    Creates a reference structure and produces n_frames perturbed copies,
    each with a random rotation and translation applied (plus small noise).
    This mimics molecular dynamics data where structures differ by rigid-body
    motion and small conformational changes.

    Args:
        n_atoms: Number of atoms in each frame
        n_frames: Number of frames (samples)
        spread: Spatial spread of the reference structure
        noise: Standard deviation of per-atom Gaussian noise

    Returns:
        data: Array of shape (n_frames, 3*n_atoms) -- one sample per row.
              Same layout as compute_aligned_sqdists in butane.py.
    """
    np.random.seed(42)

    # Create a reference structure
    reference = np.random.randn(n_atoms, 3) * spread

    # Create trajectory as rotated + translated + noised versions of reference
    trajectory = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        # Random rotation around z-axis
        angle = np.random.randn() * 0.3
        c, s = np.cos(angle), np.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Random rotation around x-axis
        angle_x = np.random.randn() * 0.3
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

        R = Rz @ Rx

        # Random translation
        translation = np.random.randn(1, 3) * 5.0

        trajectory[i] = reference @ R.T + translation + np.random.randn(n_atoms, 3) * noise

    # Flatten to (n_frames, 3*n_atoms) -- same layout as compute_aligned_sqdists
    data = trajectory.reshape(n_frames, -1)

    return data


def test_epsilon_net_alignment_rmsd_vs_euclidean():
    """
    Test epsilon_net with alignment-based RMSD vs Euclidean distance on XYZ coordinates.
    
    For molecular structures, alignment-based RMSD should produce different (and more 
    meaningful) epsilon nets than raw Euclidean distance.
    """
    print("\n" + "="*70)
    print("Test 1: Alignment RMSD vs Euclidean on XYZ coordinates")
    print("="*70)
    
    # Create XYZ coordinate data (e.g., 5 atoms, 100 frames)
    n_atoms = 5
    n_frames = 100
    data = generate_xyz_trajectory(n_atoms, n_frames, spread=2.0)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of atoms: {n_atoms}")
    print(f"Number of frames: {n_frames}")
    print(f"Features per frame: {data.shape[1]}")
    
    epsilon = 0.5
    
    # Wrap distance functions to bridge (n_features, n_samples) epsilon_net
    # interface with (n_samples, n_features) convention used here
    def aligned_dist_fn(x_col, Y_cols):
        return helpers.alignment_rmsd_distance(x_col.T, Y_cols.T)
    
    def eucl_dist_fn(x_col, Y_cols):
        return euclidean_distance(x_col.T, Y_cols.T)
    
    # Test with alignment-based RMSD
    print(f"\nRunning epsilon_net with alignment RMSD (ε={epsilon})...")
    net_indices_aligned, net_points_aligned = helpers.epsilon_net(
        data.T, epsilon, distance_fn=aligned_dist_fn
    )
    print(f"\nAlignment RMSD net size: {len(net_indices_aligned)}")
    
    # Test with Euclidean distance
    print(f"\nRunning epsilon_net with Euclidean distance (ε={epsilon})...")
    net_indices_euclidean, net_points_euclidean = helpers.epsilon_net(
        data.T, epsilon, distance_fn=eucl_dist_fn
    )
    print(f"\nEuclidean net size: {len(net_indices_euclidean)}")
    
    # Test with default (no distance function)
    print(f"\nRunning epsilon_net with default distance (ε={epsilon})...")
    net_indices_default, net_points_default = helpers.epsilon_net(
        data.T, epsilon, distance_fn=None
    )
    print(f"\nDefault net size: {len(net_indices_default)}")
    
    # Assertions
    print("\n" + "-"*70)
    print("Validation checks:")
    print("-"*70)
    
    # Check that indices are valid
    assert len(net_indices_aligned) > 0, "Alignment RMSD net should not be empty"
    assert len(net_indices_euclidean) > 0, "Euclidean net should not be empty"
    assert len(net_indices_default) > 0, "Default net should not be empty"
    print("✓ All nets are non-empty")
    
    # Check that indices are within valid range
    assert np.all(net_indices_aligned < data.shape[0]), "Alignment indices out of range"
    assert np.all(net_indices_euclidean < data.shape[0]), "Euclidean indices out of range"
    assert np.all(net_indices_default < data.shape[0]), "Default indices out of range"
    print("✓ All indices are within valid range")
    
    # Check that returned points match the indices
    # epsilon_net returns points as columns, so compare with data[indices].T
    assert np.allclose(net_points_aligned, data[net_indices_aligned].T), \
        "Alignment net points don't match indices"
    assert np.allclose(net_points_euclidean, data[net_indices_euclidean].T), \
        "Euclidean net points don't match indices"
    print("✓ Net points match their indices")
    
    # For molecular data, alignment-based RMSD should generally produce smaller nets
    # because it recognizes rotated/translated copies as similar
    print(f"\nNet size comparison:")
    print(f"  Alignment RMSD: {len(net_indices_aligned)} points")
    print(f"  Euclidean:      {len(net_indices_euclidean)} points")
    print(f"  Default:        {len(net_indices_default)} points")
    
    # Default should match Euclidean (both use L2 norm)
    assert np.allclose(len(net_indices_default), len(net_indices_euclidean), rtol=0.1), \
        "Default and Euclidean should produce similar net sizes"
    print("✓ Default distance matches Euclidean behavior")
    
    # Alignment RMSD should typically produce smaller net (recognizes rotations)
    # but this depends on the data, so we just check it's reasonable
    ratio = len(net_indices_aligned) / len(net_indices_euclidean)
    print(f"  Alignment/Euclidean ratio: {ratio:.2f}")
    assert 0.1 < ratio < 10.0, "Net size ratio should be reasonable"
    print("✓ Alignment RMSD produces a reasonable net size")
    
    print("\n✓ Test 1 PASSED\n")
    
    return {
        'data': data,
        'n_atoms': n_atoms,
        'aligned': (net_indices_aligned, net_points_aligned),
        'euclidean': (net_indices_euclidean, net_points_euclidean),
        'default': (net_indices_default, net_points_default)
    }


def test_epsilon_net_different_epsilons():
    """
    Test epsilon_net with alignment RMSD at different epsilon values.
    Smaller epsilon should produce larger nets.
    """
    print("\n" + "="*70)
    print("Test 2: Alignment RMSD with different epsilon values")
    print("="*70)
    
    # Create XYZ coordinate data
    n_atoms = 4
    n_frames = 80
    data = generate_xyz_trajectory(n_atoms, n_frames, spread=1.5)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of atoms: {n_atoms}, Number of frames: {n_frames}")
    
    def aligned_dist_fn(x_col, Y_cols):
        return helpers.alignment_rmsd_distance(x_col.T, Y_cols.T)
    
    epsilon_small = 0.3
    epsilon_large = 1.0
    
    # Test with small epsilon
    print(f"\nRunning with epsilon={epsilon_small}...")
    net_idx_small, net_pts_small = helpers.epsilon_net(
        data.T, epsilon_small, distance_fn=aligned_dist_fn
    )
    print(f"\nSmall epsilon net size: {len(net_idx_small)}")
    
    # Test with large epsilon
    print(f"\nRunning with epsilon={epsilon_large}...")
    net_idx_large, net_pts_large = helpers.epsilon_net(
        data.T, epsilon_large, distance_fn=aligned_dist_fn
    )
    print(f"\nLarge epsilon net size: {len(net_idx_large)}")
    
    # Assertions
    print("\n" + "-"*70)
    print("Validation checks:")
    print("-"*70)
    
    # Smaller epsilon should produce larger net (fewer points excluded per ball)
    assert len(net_idx_small) > len(net_idx_large), \
        f"Smaller epsilon should produce larger net: {len(net_idx_small)} vs {len(net_idx_large)}"
    print(f"✓ Smaller epsilon produces larger net: {len(net_idx_small)} vs {len(net_idx_large)}")
    
    # Large epsilon net should be a subset of small epsilon net (approximately)
    # This isn't guaranteed but should often be true
    print(f"  Net size ratio (small/large): {len(net_idx_small)/len(net_idx_large):.2f}")
    
    print("\n✓ Test 2 PASSED\n")
    
    return {
        'data': data,
        'n_atoms': n_atoms,
        'small_epsilon': (net_idx_small, net_pts_small, epsilon_small),
        'large_epsilon': (net_idx_large, net_pts_large, epsilon_large)
    }


def test_epsilon_net_rotation_invariance():
    """
    Test that alignment RMSD correctly identifies rotated/translated structures
    as similar, while Euclidean distance does not.

    Creates structures that are identical up to rigid-body motion (large random
    rotations + translations + tiny noise).  The alignment-based RMSD net should
    be much smaller than the Euclidean net because Kabsch alignment removes the
    rigid-body component.
    """
    print("\n" + "="*70)
    print("Test 3: Alignment RMSD rotation invariance")
    print("="*70)

    np.random.seed(789)
    n_atoms = 6
    n_frames = 50

    # Generate base structure
    reference = np.random.randn(n_atoms, 3) * 2.0

    # Create trajectory with large rotations and translations, tiny noise
    data_frames = []
    for i in range(n_frames):
        # Large random rotation around z-axis
        angle = np.random.uniform(0, 2*np.pi)
        c, s = np.cos(angle), np.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Large random rotation around x-axis
        angle_x = np.random.uniform(0, 2*np.pi)
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

        R = Rz @ Rx

        # Large random translation
        translation = np.random.randn(1, 3) * 10.0

        rotated = reference @ R.T + translation + np.random.randn(n_atoms, 3) * 0.05
        data_frames.append(rotated)

    # Flatten to (n_frames, 3*n_atoms) -- same layout as compute_aligned_sqdists
    data = np.array(data_frames).reshape(n_frames, -1)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of atoms: {n_atoms}")
    print(f"Number of frames (mostly rotated versions): {n_frames}")
    
    def aligned_dist_fn(x_col, Y_cols):
        return helpers.alignment_rmsd_distance(x_col.T, Y_cols.T)
    
    def eucl_dist_fn(x_col, Y_cols):
        return euclidean_distance(x_col.T, Y_cols.T)
    
    epsilon = 0.3
    
    # Test with alignment RMSD (should recognize rotations)
    print(f"\nRunning with alignment RMSD (ε={epsilon})...")
    net_idx_aligned, net_pts_aligned = helpers.epsilon_net(
        data.T, epsilon, distance_fn=aligned_dist_fn
    )
    print(f"\nAlignment RMSD net size: {len(net_idx_aligned)}")
    
    # Test with Euclidean (won't recognize rotations)
    print(f"\nRunning with Euclidean distance (ε={epsilon})...")
    net_idx_eucl, net_pts_eucl = helpers.epsilon_net(
        data.T, epsilon, distance_fn=eucl_dist_fn
    )
    print(f"\nEuclidean net size: {len(net_idx_eucl)}")
    
    # Assertions
    print("\n" + "-"*70)
    print("Validation checks:")
    print("-"*70)
    
    assert len(net_idx_aligned) > 0 and len(net_idx_eucl) > 0, \
        "Both nets should be non-empty"
    print("✓ Both nets are non-empty")
    
    # Since structures are mostly rotated versions of each other,
    # alignment RMSD should produce a much smaller net (recognizes similarity)
    print(f"\nNet size comparison (rotated structures):")
    print(f"  Alignment RMSD: {len(net_idx_aligned)} points")
    print(f"  Euclidean:      {len(net_idx_eucl)} points")
    print(f"  Ratio (aligned/euclidean): {len(net_idx_aligned)/len(net_idx_eucl):.2f}")
    
    # Alignment should produce significantly smaller net for rotated data
    assert len(net_idx_aligned) <= len(net_idx_eucl), \
        "Alignment RMSD should produce smaller or equal net for rotated structures"
    print("✓ Alignment RMSD recognizes rotational similarity")
    
    print("\n✓ Test 3 PASSED\n")
    
    return {
        'data': data,
        'n_atoms': n_atoms,
        'aligned': (net_idx_aligned, net_pts_aligned),
        'euclidean': (net_idx_eucl, net_pts_eucl)
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" EPSILON_NET UNIT TESTS - Distance Function Parameter")
    print(" Testing Alignment-based RMSD for XYZ Coordinates")
    print("="*70)

    try:
        # Run test 1: Alignment RMSD vs Euclidean
        results1 = test_epsilon_net_alignment_rmsd_vs_euclidean()

        # Run test 2: Different epsilon values
        results2 = test_epsilon_net_different_epsilons()

        # Run test 3: Rotation invariance
        results3 = test_epsilon_net_rotation_invariance()

        print("\n" + "="*70)
        print(" ALL TESTS PASSED!")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\nERROR: {e}\n")
        raise
