#!/usr/bin/env python
"""
Phase 2-7 MDM_evaluation 
------------------------------------------
Fixes:
1) JSON serialization: convert tuple keys to strings
2) File I/O: ensure reading the corrected outputs
3) Numeric range validation: add consistency checks
4) Improved metrics: more reliable quality evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class CUB15Evaluator:
    """CUB-15 sequence quality evaluator (fixed)"""
    
    def __init__(self):
        # CUB-15 skeleton connections
        self.connections = [
            (0, 1), (1, 6), (6, 7), (7, 8),      # beak→crown→nape→back→breast
            (1, 2), (2, 5),                      # crown→forehead→throat
            (1, 3), (1, 4),                      # crown→eyes
            (8, 9),                              # breast→belly
            (7, 12),                             # back→tail
            (7, 10), (7, 11),                    # back→wings
            (9, 13), (9, 14),                    # belly→legs
        ]
        
        self.joint_names = [
            'beak', 'crown', 'forehead', 'left_eye', 'right_eye',      # 0-4
            'throat', 'nape', 'back', 'breast', 'belly',               # 5-9
            'left_wing', 'right_wing', 'tail',                         # 10-12
            'left_leg', 'right_leg'                                    # 13-14
        ]
        
        # Reasonable bone-length ranges (adjusted from real CUB data)
        self.bone_length_ranges = {
            (0, 1): (0.05, 0.4),   # beak-crown
            (1, 6): (0.1, 0.5),    # crown-nape
            (6, 7): (0.2, 0.8),    # nape-back
            (7, 8): (0.1, 0.6),    # back-breast
            (8, 9): (0.1, 0.5),    # breast-belly
            (7, 10): (0.3, 2.0),   # back-left_wing (wings can be longer)
            (7, 11): (0.3, 2.0),   # back-right_wing
            (7, 12): (0.1, 0.8),   # back-tail
        }
        
        # Connection name map (for JSON serialization)
        self.connection_names = {}
        for joint1, joint2 in self.connections:
            name1 = self.joint_names[joint1]
            name2 = self.joint_names[joint2]
            self.connection_names[(joint1, joint2)] = f"{name1}-{name2}"
    
    def _validate_input_data(self, sequence):
        """Validate input sequence consistency."""
        print("Data validation:")
        print(f"  Input shape: {sequence.shape}")
        
        if len(sequence.shape) != 3:
            raise ValueError(f"Sequence shape must be (T,J,C), got {sequence.shape}")
        
        T, J, C = sequence.shape
        if J != 15:
            raise ValueError(f"Expected 15 joints, got {J}")
        if C != 3:
            raise ValueError(f"Expected 3 channels (x,y,vis), got {C}")
        
        coords = sequence[:, :, :2]
        visibility = sequence[:, :, 2]
        
        coord_range = [coords.min(), coords.max()]
        print(f"  Coord range: [{coord_range[0]:.3f}, {coord_range[1]:.3f}]")
        print(f"  Visibility range: [{visibility.min():.3f}, {visibility.max():.3f}]")
        
        # Outlier checks
        if abs(coord_range[0]) > 100 or abs(coord_range[1]) > 100:
            print("  Warning: unusually large coordinate range; data may be problematic")
        
        if np.isnan(sequence).any():
            print("  ERROR: NaN detected")
        if np.isinf(sequence).any():
            print("  ERROR: Inf detected")
        
        return True
    
    def evaluate_sequence(self, sequence, verbose=True):
        """
        Evaluate a single sequence.
        
        Args:
            sequence: numpy array (T, J, C) — time × joints × channels
            verbose: print details
        
        Returns:
            dict: evaluation results (JSON-serializable)
        """
        if verbose:
            print(f"Evaluating sequence: {sequence.shape}")
        
        self._validate_input_data(sequence)
        
        results = {}
        results['basic'] = self._check_basic_stats(sequence, verbose)
        results['skeleton'] = self._check_skeleton_validity(sequence, verbose)
        results['temporal'] = self._check_temporal_consistency(sequence, verbose)
        results['motion'] = self._check_motion_patterns(sequence, verbose)
        results['overall'] = self._compute_overall_score(results, verbose)
        
        results = self._ensure_json_serializable(results)
        return results
    
    def _check_basic_stats(self, sequence, verbose):
        """Basic numerical statistics."""
        T, J, C = sequence.shape
        coords = sequence[:, :, :2]
        visibility = sequence[:, :, 2]
        
        stats = {
            'shape': list(sequence.shape),
            'coord_range': [float(coords.min()), float(coords.max())],
            'has_nan': bool(np.isnan(sequence).any()),
            'has_inf': bool(np.isinf(sequence).any()),
            'avg_visibility': float(np.mean(visibility)),
            'coord_std': float(np.std(coords)),
            'coord_mean': [float(coords[:,:,0].mean()), float(coords[:,:,1].mean())],
        }
        
        if verbose:
            print("  Basic stats:")
            print(f"    Coord range: [{stats['coord_range'][0]:.3f}, {stats['coord_range'][1]:.3f}]")
            print(f"    Avg visibility: {stats['avg_visibility']:.3f}")
            print(f"    Coord std: {stats['coord_std']:.3f}")
            if stats['has_nan']:
                print("    ERROR: NaN detected")
            if stats['has_inf']:
                print("    ERROR: Inf detected")
        
        return stats
    
    def _check_skeleton_validity(self, sequence, verbose):
        """Check validity of skeleton connections."""
        T, J, C = sequence.shape
        
        # Use string keys (not tuple) for JSON friendliness
        bone_lengths = {}
        bone_length_stds = {}
        violations = 0
        total_checks = len(self.bone_length_ranges)
        
        for joint1, joint2 in self.connections:
            # Compute bone lengths per frame
            lengths = []
            for t in range(T):
                try:
                    dist = euclidean(sequence[t, joint1, :2], sequence[t, joint2, :2])
                    lengths.append(dist)
                except:
                    lengths.append(0.0)
            
            bone_key = self.connection_names.get((joint1, joint2), f"joint_{joint1}_joint_{joint2}")
            bone_lengths[bone_key] = float(np.mean(lengths))
            bone_length_stds[bone_key] = float(np.std(lengths))
            
            # Range check if available
            if (joint1, joint2) in self.bone_length_ranges:
                min_len, max_len = self.bone_length_ranges[(joint1, joint2)]
                avg_len = bone_lengths[bone_key]
                
                # Scale range based on overall coordinate magnitude (rough heuristic)
                expected_scale = max(1.0, abs(sequence[:,:,:2]).max())
                scaled_min = min_len * expected_scale
                scaled_max = max_len * expected_scale
                
                if not (scaled_min <= avg_len <= scaled_max):
                    violations += 1
        
        skeleton_stats = {
            'bone_lengths': bone_lengths,
            'bone_length_stds': bone_length_stds,
            'length_violations': violations,
            'total_bones_checked': total_checks,
            'violation_rate': float(violations / total_checks) if total_checks > 0 else 0.0,
            'avg_bone_stability': float(np.mean(list(bone_length_stds.values()))) if bone_length_stds else 0.0,
        }
        
        if verbose:
            print("  Skeleton check:")
            print(f"    Length violations: {violations}/{total_checks} ({skeleton_stats['violation_rate']:.1%})")
            print(f"    Avg bone-length stability (std): {skeleton_stats['avg_bone_stability']:.4f}")
            key_bones = [(0, 1), (7, 10), (7, 11)]  # beak-crown, back-left/right_wing
            for bone in key_bones:
                bone_key = self.connection_names.get(bone, f"joint_{bone[0]}_joint_{bone[1]}")
                if bone_key in bone_lengths:
                    length = bone_lengths[bone_key]
                    std = bone_length_stds[bone_key]
                    print(f"    {bone_key}: {length:.3f}±{std:.3f}")
        
        return skeleton_stats
    
    def _check_temporal_consistency(self, sequence, verbose):
        """Check temporal consistency."""
        T, J, C = sequence.shape
        
        # Frame-to-frame displacement
        frame_diffs = []
        for t in range(1, T):
            try:
                diff = np.mean(np.linalg.norm(
                    sequence[t, :, :2] - sequence[t-1, :, :2], axis=1
                ))
                frame_diffs.append(diff)
            except:
                frame_diffs.append(0.0)
        
        if not frame_diffs:
            frame_diffs = [0.0]
        
        frame_diffs = np.array(frame_diffs)
        if len(frame_diffs) > 1:
            diff_threshold = np.mean(frame_diffs) + 2 * np.std(frame_diffs)
            jumps = np.sum(frame_diffs > diff_threshold)
        else:
            jumps = 0
        
        # Smoothness via second difference
        smoothness_scores = []
        for j in range(J):
            for c in range(2):  # x, y
                trajectory = sequence[:, j, c]
                if len(trajectory) > 2:
                    accel = np.diff(trajectory, n=2)
                    if len(accel) > 0:
                        smoothness = 1.0 / (1.0 + np.std(accel))
                        smoothness_scores.append(smoothness)
        
        if not smoothness_scores:
            smoothness_scores = [0.0]
        
        temporal_stats = {
            'avg_frame_diff': float(np.mean(frame_diffs)),
            'frame_diff_std': float(np.std(frame_diffs)),
            'sudden_jumps': int(jumps),
            'avg_smoothness': float(np.mean(smoothness_scores)),
            'total_frames': T,
        }
        
        if verbose:
            print("  Temporal check:")
            print(f"    Avg frame-to-frame displacement: {temporal_stats['avg_frame_diff']:.4f}")
            print(f"    Sudden jumps: {temporal_stats['sudden_jumps']}")
            print(f"    Avg smoothness: {temporal_stats['avg_smoothness']:.4f}")
        
        return temporal_stats
    
    def _check_motion_patterns(self, sequence, verbose):
        """Check motion pattern features."""
        T, J, C = sequence.shape
        
        try:
            # Wing motion
            left_wing_motion = np.std(sequence[:, 10, :2], axis=0)
            right_wing_motion = np.std(sequence[:, 11, :2], axis=0)
            
            # Symmetry (x-axis as proxy)
            wing_motion_diff = abs(left_wing_motion[0] - right_wing_motion[0])
            wing_motion_sum = left_wing_motion[0] + right_wing_motion[0] + 1e-8
            wing_symmetry = 1.0 - wing_motion_diff / wing_motion_sum
            wing_symmetry = max(0.0, min(1.0, wing_symmetry))
            
            # Body center stability
            body_joints = [6, 7, 8, 9]  # nape, back, breast, belly
            body_center = np.mean(sequence[:, body_joints, :2], axis=1)  # (T, 2)
            body_stability = 1.0 / (1.0 + np.std(body_center, axis=0).mean())
            
            # Overall motion amplitude
            overall_motion = np.std(sequence[:, :, :2].reshape(T, -1), axis=0).mean()
            
        except Exception as e:
            print(f"Motion pattern computation error: {e}")
            wing_symmetry = 0.0
            body_stability = 0.0
            overall_motion = 0.0
            left_wing_motion = np.array([0.0, 0.0])
            right_wing_motion = np.array([0.0, 0.0])
        
        motion_stats = {
            'wing_symmetry': float(wing_symmetry),
            'body_stability': float(body_stability),
            'overall_motion': float(overall_motion),
            'left_wing_activity': float(np.mean(left_wing_motion)),
            'right_wing_activity': float(np.mean(right_wing_motion)),
        }
        
        if verbose:
            print("  Motion patterns:")
            print(f"    Wing symmetry: {motion_stats['wing_symmetry']:.4f}")
            print(f"    Body stability: {motion_stats['body_stability']:.4f}")
            print(f"    Overall motion: {motion_stats['overall_motion']:.4f}")
        
        return motion_stats
    
    def _compute_overall_score(self, results, verbose):
        """Compute overall score."""
        scores = {}
        
        # Basic score (0–1)
        basic = results['basic']
        scores['basic'] = 1.0 - (0.5 if basic['has_nan'] else 0) - (0.5 if basic['has_inf'] else 0)
        
        # Skeleton score
        skeleton = results['skeleton']
        violation_rate = skeleton['violation_rate']
        stability_score = min(1.0, 1.0 / (1.0 + skeleton['avg_bone_stability']))
        scores['skeleton'] = max(0.0, (1.0 - violation_rate) * stability_score)
        
        # Temporal score
        temporal = results['temporal']
        jump_penalty = min(0.5, temporal['sudden_jumps'] * 0.1)
        smoothness_score = temporal['avg_smoothness']
        scores['temporal'] = max(0.0, min(1.0, smoothness_score - jump_penalty))
        
        # Motion score
        motion = results['motion']
        scores['motion'] = (motion['wing_symmetry'] + min(1.0, motion['body_stability'])) / 2
        
        # Adaptive weights
        if basic['has_nan'] or basic['has_inf']:
            weights = {'basic': 0.4, 'skeleton': 0.2, 'temporal': 0.2, 'motion': 0.2}
        else:
            weights = {'basic': 0.1, 'skeleton': 0.3, 'temporal': 0.3, 'motion': 0.3}
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        overall_score = max(0.0, min(1.0, overall_score))
        
        overall_stats = {
            'component_scores': scores,
            'weights': weights,
            'overall_score': float(overall_score),
            'grade': self._score_to_grade(overall_score)
        }
        
        if verbose:
            print("  Overall score:")
            for component, score in scores.items():
                print(f"    {component}: {score:.3f}")
            print(f"    Total: {overall_score:.3f} ({overall_stats['grade']})")
        
        return overall_stats
    
    def _score_to_grade(self, score):
        """Map score to grade string."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Pass"
        else:
            return "Fail"
    
    def _ensure_json_serializable(self, obj):
        """Ensure data is JSON-serializable."""
        if isinstance(obj, dict):
            return {str(k): self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def evaluate_batch(self, sequences, save_path=None):
        """Evaluate a batch of sequences."""
        print(f"Batch evaluation for {len(sequences)} sequences")
        
        all_results = []
        scores = []
        
        for i, seq in enumerate(sequences):
            print(f"\nSequence {i+1}/{len(sequences)}:")
            try:
                result = self.evaluate_sequence(seq, verbose=True)
                all_results.append(result)
                scores.append(result['overall']['overall_score'])
            except Exception as e:
                print(f"ERROR: sequence {i+1} evaluation failed: {e}")
                failed_result = {
                    'error': str(e),
                    'overall': {'overall_score': 0.0, 'grade': 'Error'}
                }
                all_results.append(failed_result)
                scores.append(0.0)
        
        valid_scores = [s for s in scores if s > 0]
        
        batch_stats = {
            'count': len(sequences),
            'valid_count': len(valid_scores),
            'scores': scores,
            'avg_score': float(np.mean(valid_scores)) if valid_scores else 0.0,
            'std_score': float(np.std(valid_scores)) if len(valid_scores) > 1 else 0.0,
            'min_score': float(np.min(valid_scores)) if valid_scores else 0.0,
            'max_score': float(np.max(valid_scores)) if valid_scores else 0.0,
            'grade_distribution': {}
        }
        
        grades = []
        for result in all_results:
            if 'overall' in result and 'grade' in result['overall']:
                grades.append(result['overall']['grade'])
            else:
                grades.append('Error')
                
        for grade in set(grades):
            batch_stats['grade_distribution'][grade] = grades.count(grade)
        
        print("\nBatch summary:")
        print(f"  Valid sequences: {batch_stats['valid_count']}/{batch_stats['count']}")
        if valid_scores:
            print(f"  Mean score: {batch_stats['avg_score']:.3f} ± {batch_stats['std_score']:.3f}")
            print(f"  Score range: [{batch_stats['min_score']:.3f}, {batch_stats['max_score']:.3f}]")
        print(f"  Grade distribution: {batch_stats['grade_distribution']}")
        
        if save_path:
            output = {
                'individual_results': all_results,
                'batch_statistics': batch_stats,
                'evaluation_metadata': {
                    'evaluator_version': 'CUB15Evaluator_v2_fixed',
                    'total_sequences': len(sequences),
                    'evaluation_time': str(np.datetime64('now'))
                }
            }
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Saved batch evaluation to: {save_path}")
            except Exception as e:
                print(f"ERROR: failed to save: {e}")
        
        return all_results, batch_stats

def load_sequence_with_validation(file_path):
    """Safely load a sequence file with validation."""
    print(f"Loading sequence file: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = np.load(file_path)
        print(f"  Loaded. Shape: {data.shape}")
        
        # Range check
        if len(data.shape) >= 3:
            coords = data[..., :2]
            coord_range = [coords.min(), coords.max()]
            print(f"  Coord range: [{coord_range[0]:.3f}, {coord_range[1]:.3f}]")
            if abs(coord_range[0]) > 1000 or abs(coord_range[1]) > 1000:
                print("  Warning: unusually large coordinate range; values may be unstable")
        
        return data
        
    except Exception as e:
        print(f"  ERROR: load failed: {e}")
        raise

def main():
    """Entry point: demo evaluation."""
    print("CUB-15 Sequence Quality Evaluator (Fixed)")
    print("=" * 60)
    
    evaluator = CUB15Evaluator()
    
    # Prefer the fixed-version files if present
    sequence_files = [
        "generated_pose_seq_fixed.npy",
        "generated_batch_sequences_fixed.npy",
        "generated_pose_seq.npy",
        "generated_batch_sequences.npy"
    ]
    
    found_files = []
    for f in sequence_files:
        if Path(f).exists():
            found_files.append(f)
            print(f"Found file: {f}")
    
    if not found_files:
        print("No generated sequence files found.")
        print("Please run 2.4-MDM_inference_CUB15_fixed.py first to generate sequences.")
        return
    
    # Evaluate single sequence
    single_files = [f for f in found_files if 'batch' not in f]
    if single_files:
        print("\nEvaluate single sequence")
        sequence_file = single_files[0]
        print(f"Using file: {sequence_file}")
        
        try:
            sequence = load_sequence_with_validation(sequence_file)
            result = evaluator.evaluate_sequence(sequence, verbose=True)
            
            output_file = "single_sequence_evaluation_fixed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved single-sequence evaluation: {output_file}")
            
        except Exception as e:
            print(f"ERROR: single-sequence evaluation failed: {e}")
    
    # Evaluate batch sequence(s)
    batch_files = [f for f in found_files if 'batch' in f]
    if batch_files:
        print("\nEvaluate batch sequences")
        batch_file = batch_files[0]
        print(f"Using file: {batch_file}")
        
        try:
            batch_sequences = load_sequence_with_validation(batch_file)
            print(f"Batch shape: {batch_sequences.shape}")
            
            sequences_list = [batch_sequences[i] for i in range(len(batch_sequences))]
            
            individual_results, batch_stats = evaluator.evaluate_batch(
                sequences_list, 
                save_path="batch_sequences_evaluation.json"
            )
            
        except Exception as e:
            print(f"ERROR: batch evaluation failed: {e}")
    
    print("\nEvaluation finished.")
    print("Output files:")
    if single_files:
        print("  - single_sequence_evaluation_fixed.json")
    if batch_files:
        print("  - batch_sequences_evaluation.json")
    
    print("\nNotes:")
    print("  - If coordinate ranges are abnormal, check how sequences were saved in 2.4.")
    print("  - Overall score > 0.7 generally indicates good quality.")
    print("  - Use component scores to diagnose issues.")

if __name__ == "__main__":
    main()
