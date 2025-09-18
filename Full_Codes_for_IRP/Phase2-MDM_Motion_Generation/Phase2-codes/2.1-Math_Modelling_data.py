#!/usr/bin/env python
"""
Phase 2-1: CUB-15 Bird Behavior Data Generator (Enhanced v5.1 – Large Wing-Flap + Training Labels)
A complete bird behavior library based on realistic biomechanical modeling.

New features:
1. Full ground behaviors (hopping, walking, head turns, etc.)
2. Enhanced flight motions (large-amplitude flaps, sharp turns, etc.)
3. Composite behavior sequences (ground → takeoff → flight → landing)
4. More realistic physical constraints and motion transitions
5. Large-amplitude wing flapping effects
6. More static folded-wing poses
7. Automatic generation and saving of six-class training labels
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean

# ===== NEW: canonical labels mapping =====
CANONICAL_LABELS = ["takeoff","gliding","hovering","soaring","diving","landing"]
LABEL_TO_ID = {n: i for i, n in enumerate(CANONICAL_LABELS)}

PHASE_TO_CANON = {
    # Transitions (takeoff/landing)
    "prepare_takeoff": "takeoff",
    "takeoff":         "takeoff",
    "landing_approach":"landing",
    "landing":         "landing",
    # Flight
    "hover":           "hovering",
    "glide":           "gliding",
    "soar":            "soaring",
    "cruise":          "gliding",       # Map cruise to gliding
    "power_flap":      "gliding",       # Strong flaps but not hovering → gliding
    "climb":           "soaring",       # Climb approximated as soaring/slow ascent
    "dive":            "diving",
    "turn_left":       "gliding",
    "turn_right":      "gliding",
    "sharp_turn_left": "gliding",
    "sharp_turn_right":"gliding",
    "barrel_roll":     "gliding",
    # Ground (not counted in the six classes)
    "standing":               None,
    "head_turn_left":         None,
    "head_turn_right":        None,
    "hop_forward":            None,
    "hop_left":               None,
    "hop_right":              None,
    "walking":                None,
    "pecking":                None,
    "preening":               None,
    "side_hop_left":          None,
    "side_hop_right":         None,
    "folded_wing_hop":        None,
    "diagonal_hop":           None,
    "side_walk":              None,
    "crouch_hop":             None,
    "zigzag_hop":             None,
    "backward_hop":           None,
    "side_turn_hop":          None,
    "wing_tuck_walk":         None,
    "static_folded_standing": None,
    "one_leg_folded_rest":    None,
    "folded_observing":       None,
    "relaxed_folded_stand":   None,
}

@dataclass
class BirdSpecies:
    """Bird species parameters (based on real biological data)"""
    name: str
    wingspan: float          # Wingspan scale
    body_length: float       # Body length
    wing_loading: float      # Wing loading (weight/wing area)
    flap_frequency: float    # Base flapping frequency (Hz)
    glide_ratio: float       # Gliding preference
    muscle_power: float      # Muscle power coefficient
    agility: float           # Maneuverability coefficient
    hop_ability: float       # Hopping ability
    ground_preference: float # Ground activity preference

# Extended species set
BIRD_SPECIES = [
    BirdSpecies("eagle", 2.2, 1.3, 0.8, 0.7, 0.8, 1.2, 0.6, 0.3, 0.2),       # Large raptor
    BirdSpecies("sparrow", 1.0, 0.8, 0.3, 2.8, 0.1, 0.8, 1.0, 0.9, 0.8),     # Small passerine
    BirdSpecies("seagull", 1.8, 1.1, 0.5, 1.1, 0.6, 1.0, 0.8, 0.4, 0.3),     # Medium seabird
    BirdSpecies("hummingbird", 0.7, 0.6, 0.2, 12.0, 0.0, 1.5, 1.2, 0.2, 0.1),# Tiny high-frequency
    BirdSpecies("albatross", 2.8, 1.5, 1.2, 0.4, 0.95, 0.9, 0.4, 0.1, 0.05), # Glide specialist
    BirdSpecies("falcon", 1.4, 1.0, 0.6, 1.8, 0.4, 1.3, 1.1, 0.5, 0.3),      # Speed type
    BirdSpecies("robin", 1.1, 0.9, 0.35, 2.5, 0.2, 0.9, 0.8, 0.8, 0.7),      # Ground-active
    BirdSpecies("crow", 1.6, 1.2, 0.6, 1.5, 0.3, 1.1, 0.7, 0.7, 0.6),        # Intelligent corvid
]

@dataclass
class BehaviorPhase:
    """Behavior phase definition (extended – includes ground and flight)"""
    name: str
    category: str                       # 'ground', 'aerial', 'transition'
    duration_range: Tuple[int, int]     # Frame count range
    wing_amplitude: float               # Wing amplitude
    wing_frequency_mult: float          # Wing frequency multiplier
    body_tilt: float                    # Body tilt angle
    tail_angle: float                   # Tail angle
    leg_retraction: float               # Leg retraction (negative = extended)
    energy_level: float                 # Energy consumption level
    height_change: float                # Tendency of altitude change
    movement_speed: float               # Movement speed

# Full behavior phase library
BEHAVIOR_PHASES = [
    # === Ground behaviors ===
    BehaviorPhase("standing", "ground", (8, 16), 0.1, 0.1, 0.0, 0.0, -0.8, 0.1, 0.0, 0.0),
    BehaviorPhase("head_turn_left", "ground", (6, 12), 0.1, 0.1, 0.0, 0.0, -0.8, 0.1, 0.0, 0.0),
    BehaviorPhase("head_turn_right", "ground", (6, 12), 0.1, 0.1, 0.0, 0.0, -0.8, 0.1, 0.0, 0.0),
    BehaviorPhase("hop_forward", "ground", (4, 8), 0.3, 0.5, -0.1, 0.1, -0.6, 0.6, 0.2, 0.4),
    BehaviorPhase("hop_left", "ground", (4, 8), 0.3, 0.5, 0.1, -0.2, -0.6, 0.6, 0.1, 0.3),
    BehaviorPhase("hop_right", "ground", (4, 8), 0.3, 0.5, 0.1, 0.2, -0.6, 0.6, 0.1, 0.3),
    BehaviorPhase("walking", "ground", (12, 20), 0.2, 0.3, 0.0, 0.0, -0.7, 0.3, 0.0, 0.2),
    BehaviorPhase("pecking", "ground", (6, 12), 0.1, 0.1, -0.6, 0.0, -0.9, 0.2, -0.1, 0.0),
    BehaviorPhase("preening", "ground", (10, 18), 0.2, 0.2, 0.2, 0.3, -0.8, 0.2, 0.0, 0.0),

    # === Added: side and folded-wing actions ===
    BehaviorPhase("side_hop_left", "ground", (6, 10), 0.1, 0.1, 0.3, -0.5, -0.7, 0.7, 0.2, 0.5),
    BehaviorPhase("side_hop_right", "ground", (6, 10), 0.1, 0.1, 0.3, 0.5, -0.7, 0.7, 0.2, 0.5),
    BehaviorPhase("folded_wing_hop", "ground", (5, 9), 0.05, 0.05, 0.0, 0.0, -0.8, 0.6, 0.3, 0.4),
    BehaviorPhase("diagonal_hop", "ground", (5, 8), 0.2, 0.3, 0.2, 0.3, -0.6, 0.6, 0.2, 0.6),
    BehaviorPhase("side_walk", "ground", (10, 16), 0.1, 0.1, 0.4, -0.2, -0.8, 0.3, 0.0, 0.3),
    BehaviorPhase("crouch_hop", "ground", (4, 7), 0.1, 0.2, -0.3, 0.1, -0.9, 0.5, 0.1, 0.3),
    BehaviorPhase("zigzag_hop", "ground", (8, 14), 0.2, 0.4, 0.1, 0.0, -0.7, 0.7, 0.2, 0.4),
    BehaviorPhase("backward_hop", "ground", (4, 8), 0.1, 0.2, 0.2, 0.0, -0.7, 0.6, 0.1, -0.3),
    BehaviorPhase("side_turn_hop", "ground", (6, 10), 0.2, 0.3, 0.0, -0.6, -0.6, 0.7, 0.2, 0.2),
    BehaviorPhase("wing_tuck_walk", "ground", (12, 18), 0.05, 0.05, 0.0, 0.0, -0.9, 0.2, 0.0, 0.2),

    # === Added: static folded-wing poses ===
    BehaviorPhase("static_folded_standing", "ground", (12, 24), 0.05, 0.05, 0.0, 0.0, -0.9, 0.1, 0.0, 0.0),
    BehaviorPhase("one_leg_folded_rest", "ground", (16, 30), 0.03, 0.03, 0.1, 0.0, -0.9, 0.1, 0.0, 0.0),
    BehaviorPhase("folded_observing", "ground", (10, 20), 0.05, 0.05, 0.0, 0.1, -0.9, 0.1, 0.0, 0.0),
    BehaviorPhase("relaxed_folded_stand", "ground", (15, 25), 0.02, 0.02, 0.0, 0.0, -0.9, 0.1, 0.0, 0.0),

    # === Transition behaviors ===
    BehaviorPhase("prepare_takeoff", "transition", (6, 10), 0.4, 0.8, -0.2, 0.1, -0.5, 0.7, 0.0, 0.0),
    BehaviorPhase("takeoff", "transition", (8, 14), 1.8, 1.6, -0.4, 0.2, 0.2, 1.2, 0.8, 0.3),
    BehaviorPhase("landing_approach", "transition", (8, 14), 0.8, 1.2, -0.3, 0.2, -0.2, 0.8, -0.3, 0.1),
    BehaviorPhase("landing", "transition", (6, 12), 1.2, 1.4, -0.5, 0.4, -0.8, 0.9, -0.5, 0.0),

    # === Aerial behaviors ===
    BehaviorPhase("power_flap", "aerial", (8, 16), 2.0, 1.8, -0.1, 0.1, 0.3, 1.0, 0.2, 0.4),
    BehaviorPhase("cruise", "aerial", (16, 32), 0.8, 1.0, 0.0, 0.0, 0.2, 0.5, 0.0, 0.3),
    BehaviorPhase("glide", "aerial", (12, 24), 0.2, 0.2, 0.1, 0.05, 0.1, 0.2, 0.0, 0.2),
    BehaviorPhase("soar", "aerial", (20, 40), 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1),
    BehaviorPhase("climb", "aerial", (10, 20), 1.3, 1.3, -0.2, 0.1, 0.2, 0.9, 0.4, 0.2),
    BehaviorPhase("dive", "aerial", (8, 16), 0.6, 0.8, 0.4, 0.0, 0.3, 0.4, -0.6, 0.6),
    BehaviorPhase("turn_left", "aerial", (6, 12), 1.0, 1.1, 0.0, -0.4, 0.2, 0.7, 0.0, 0.3),
    BehaviorPhase("turn_right", "aerial", (6, 12), 1.0, 1.1, 0.0, 0.4, 0.2, 0.7, 0.0, 0.3),
    BehaviorPhase("sharp_turn_left", "aerial", (4, 8), 1.4, 1.4, -0.1, -0.6, 0.2, 0.9, 0.0, 0.2),
    BehaviorPhase("sharp_turn_right", "aerial", (4, 8), 1.4, 1.4, -0.1, 0.6, 0.2, 0.9, 0.0, 0.2),
    BehaviorPhase("hover", "aerial", (8, 16), 1.5, 2.2, -0.1, 0.0, 0.4, 1.3, 0.0, 0.0),
    BehaviorPhase("barrel_roll", "aerial", (6, 10), 1.2, 1.5, 0.0, 0.0, 0.3, 0.8, 0.0, 0.2),
]

@dataclass
class EnvironmentCondition:
    """Environmental conditions"""
    wind_speed: float       # Wind speed
    wind_direction: float   # Wind direction (radians)
    turbulence: float       # Turbulence strength
    temperature: float      # Temperature influence

class EnhancedCUBBirdGenerator:
    """Enhanced CUB-15 bird behavior generator"""

    def __init__(self):
        # CUB-200 15 joints
        self.joint_names = [
            'beak', 'crown', 'forehead', 'left_eye', 'right_eye',      # 0-4: head
            'throat', 'nape', 'back', 'breast', 'belly',               # 5-9: torso
            'left_wing', 'right_wing', 'tail',                         # 10-12: wings and tail
            'left_leg', 'right_leg'                                    # 13-14: legs
        ]

        # Skeleton connections (based on bird anatomy)
        self.connections = [
            (0, 1), (1, 6), (6, 7), (7, 8),           # main spine chain
            (1, 2), (2, 5),                           # head structure
            (1, 3), (1, 4),                           # eyes
            (8, 9),                                   # torso
            (7, 12),                                  # tail
            (7, 10), (7, 11),                         # wings
            (9, 13), (9, 14),                         # legs
        ]

        # Enhanced joint constraints
        self.joint_constraints = {
            'bone_lengths': {
                (0, 1): (0.15, 0.25),    # beak-crown
                (1, 6): (0.2, 0.3),      # crown-nape
                (6, 7): (0.4, 0.6),      # nape-back
                (7, 8): (0.3, 0.5),      # back-breast
                (8, 9): (0.2, 0.4),      # breast-belly
                (7, 10): (0.8, 1.5),     # back-left_wing
                (7, 11): (0.8, 1.5),     # back-right_wing
                (7, 12): (0.3, 0.6),     # back-tail
                (9, 13): (0.3, 0.8),     # belly-left_leg
                (9, 14): (0.3, 0.8),     # belly-right_leg
            },
            'angle_limits': {
                'neck_bend': (-1.2, 1.2),        # larger neck range
                'wing_fold': (-3.0, 0.8),        # wings can fully fold
                'tail_spread': (-0.8, 0.8),      # tail spread
                'leg_retract': (-1.5, 0.5),      # leg retraction
            }
        }

    def create_base_skeleton(self, species: BirdSpecies) -> np.ndarray:
        """Create base 15-joint skeleton layout"""
        s = species.body_length
        w = species.wingspan

        # Base posture following CUB annotation convention
        base_pose = np.array([
            # Head (0-4)
            [0.0, 1.4*s, 1.0],           # 0: beak
            [0.0, 1.3*s, 1.0],           # 1: crown
            [0.0, 1.35*s, 1.0],          # 2: forehead
            [-0.05*s, 1.32*s, 1.0],      # 3: left_eye
            [0.05*s, 1.32*s, 1.0],       # 4: right_eye

            # Torso (5-9)
            [0.0, 1.2*s, 1.0],           # 5: throat
            [0.0, 1.1*s, 1.0],           # 6: nape
            [0.0, 0.7*s, 1.0],           # 7: back
            [0.0, 0.4*s, 1.0],           # 8: breast
            [0.0, 0.0, 1.0],             # 9: belly

            # Wings and tail (10-12) – lowered wing base
            [-0.6*w, 0.3*s, 1.0],        # 10: left_wing
            [0.6*w, 0.3*s, 1.0],         # 11: right_wing
            [0.0, -0.2*s, 1.0],          # 12: tail

            # Legs (13-14)
            [-0.08*s, -0.3*s, 1.0],      # 13: left_leg
            [0.08*s, -0.3*s, 1.0],       # 14: right_leg
        ], dtype=np.float32)

        return base_pose

    def generate_behavior_sequence(self,
                                   species: BirdSpecies,
                                   phases: List[BehaviorPhase],
                                   total_frames: int = 64,
                                   environment: EnvironmentCondition = None) -> np.ndarray:
        """Generate a complete behavior sequence"""

        if environment is None:
            environment = EnvironmentCondition(0.0, 0.0, 0.0, 20.0)

        base_skeleton = self.create_base_skeleton(species)
        sequence = np.zeros((total_frames, 15, 3), dtype=np.float32)

        # Intelligent frame allocation
        frame_allocation = self._allocate_frames_intelligent(phases, total_frames)

        current_frame = 0
        current_height = 0.0  # Track current altitude

        for i, (phase, frame_count) in enumerate(zip(phases, frame_allocation)):
            end_frame = current_frame + frame_count

            # Generate phase behavior (enhanced)
            phase_sequence = self._generate_enhanced_behavior_motion(
                base_skeleton, species, phase, frame_count, environment, current_height
            )

            sequence[current_frame:end_frame] = phase_sequence

            # Update altitude
            current_height += phase.height_change * frame_count * 0.1
            current_height = max(0.0, current_height)  # Cannot go below ground

            current_frame = end_frame

        # Apply enhanced biomechanical constraints
        sequence = self._apply_enhanced_biomechanical_constraints(sequence, species)

        # Advanced smoothing
        sequence = self._advanced_smoothing(sequence)

        return sequence

    def _generate_enhanced_behavior_motion(self,
                                           base_skeleton: np.ndarray,
                                           species: BirdSpecies,
                                           phase: BehaviorPhase,
                                           frames: int,
                                           environment: EnvironmentCondition,
                                           base_height: float) -> np.ndarray:
        """Enhanced motion generator for a behavior phase"""

        motion = np.tile(base_skeleton[None, :, :], (frames, 1, 1))

        # Timeline and phase
        fps = 20.0
        t = np.linspace(0, frames / fps, frames)

        # Dynamic parameters
        base_freq = species.flap_frequency * phase.wing_frequency_mult
        wing_amp = phase.wing_amplitude * species.wingspan * 0.25

        # Wind effects
        wind_effect_x = environment.wind_speed * np.cos(environment.wind_direction) * 0.1
        wind_effect_y = environment.wind_speed * np.sin(environment.wind_direction) * 0.05

        for frame_idx in range(frames):
            current_time = t[frame_idx]
            current_pose = motion[frame_idx]
            progress = frame_idx / (frames - 1) if frames > 1 else 0

            # Primary/secondary phases
            primary_phase = 2 * np.pi * base_freq * current_time
            secondary_phase = primary_phase * 1.618

            # Base altitude offset
            height_offset = base_height + phase.height_change * progress * 2.0
            for joint_idx in range(15):
                current_pose[joint_idx, 1] += height_offset

            # Behavior-specific motion
            if phase.category == "ground":
                self._apply_ground_behavior(current_pose, phase, current_time, progress, species)
            elif phase.category == "aerial":
                self._apply_aerial_behavior(current_pose, phase, current_time, progress, species,
                                            primary_phase, wing_amp, wind_effect_x, wind_effect_y)
            elif phase.category == "transition":
                self._apply_transition_behavior(current_pose, phase, current_time, progress, species,
                                                primary_phase, wing_amp)

        return motion

    def _apply_ground_behavior(self, pose, phase: BehaviorPhase, time, progress, species: BirdSpecies):
        """Apply ground behavior"""

        if phase.name == "standing":
            body_sway = np.sin(time * 2.0) * 0.02
            pose[7:10, 0] += body_sway
            head_twitch = np.sin(time * 5.0) * 0.01
            pose[0:6, 0] += head_twitch

        elif "head_turn" in phase.name:
            direction = 1 if "right" in phase.name else -1
            turn_amount = np.sin(progress * np.pi) * 0.4 * direction
            pose[0:6, 0] += turn_amount
            pose[3, 0] += turn_amount * 0.5
            pose[4, 0] += turn_amount * 0.5

        elif phase.name == "hop_forward":
            hop_height = np.sin(progress * np.pi) * 0.4 * species.hop_ability
            hop_forward = progress * 0.3
            for joint_idx in range(15):
                pose[joint_idx, 1] += hop_height + hop_forward
            if progress < 0.3:
                leg_push = (0.3 - progress) / 0.3 * 0.2
                pose[13, 1] -= leg_push
                pose[14, 1] -= leg_push
            elif progress > 0.7:
                leg_extend = (progress - 0.7) / 0.3 * 0.3
                pose[13, 1] -= leg_extend
                pose[14, 1] -= leg_extend

        elif phase.name in ["hop_left", "hop_right"]:
            direction = -0.3 if "left" in phase.name else 0.3
            hop_height = np.sin(progress * np.pi) * 0.4 * species.hop_ability
            hop_side = progress * direction
            for joint_idx in range(15):
                pose[joint_idx, 0] += hop_side
                pose[joint_idx, 1] += hop_height
            if progress < 0.3:
                leg_push = (0.3 - progress) / 0.3 * 0.2
                pose[13, 1] -= leg_push
                pose[14, 1] -= leg_push
            elif progress > 0.7:
                leg_extend = (progress - 0.7) / 0.3 * 0.3
                pose[13, 1] -= leg_extend
                pose[14, 1] -= leg_extend

        elif phase.name == "side_hop_left":
            hop_height = np.sin(progress * np.pi) * 0.35
            side_movement = progress * -0.4
            body_lean = -0.3
            pose[6:10, 0] += body_lean
            for joint_idx in range(15):
                pose[joint_idx, 0] += side_movement
                pose[joint_idx, 1] += hop_height
            pose[10, 0] = pose[7, 0] - 0.1
            pose[11, 0] = pose[7, 0] + 0.08

        elif phase.name == "side_hop_right":
            hop_height = np.sin(progress * np.pi) * 0.35
            side_movement = progress * 0.4
            body_lean = 0.3
            pose[6:10, 0] += body_lean
            for joint_idx in range(15):
                pose[joint_idx, 0] += side_movement
                pose[joint_idx, 1] += hop_height
            pose[10, 0] = pose[7, 0] - 0.08
            pose[11, 0] = pose[7, 0] + 0.1

        elif phase.name == "folded_wing_hop":
            hop_height = np.sin(progress * np.pi) * 0.25
            random_x = np.sin(time * 7.0) * 0.1
            random_y = progress * 0.15
            for joint_idx in range(15):
                pose[joint_idx, 0] += random_x
                pose[joint_idx, 1] += hop_height + random_y
            pose[10, 0] = pose[7, 0] - 0.02
            pose[10, 1] = pose[7, 1] - 0.05
            pose[11, 0] = pose[7, 0] + 0.02
            pose[11, 1] = pose[7, 1] - 0.05

        elif phase.name == "diagonal_hop":
            hop_height = np.sin(progress * np.pi) * 0.4
            diag_x = progress * 0.25
            diag_y = progress * 0.25
            for joint_idx in range(15):
                pose[joint_idx, 0] += diag_x
                pose[joint_idx, 1] += hop_height + diag_y
            body_twist = 0.2
            pose[6:10, 0] += body_twist

        elif phase.name == "side_walk":
            step_phase = (time * 3.0) % (2 * np.pi)
            side_motion = progress * 0.4
            pose[:, 0] += side_motion
            pose[6:10, 0] += 0.4
            left_leg_side = np.sin(step_phase) * 0.15
            right_leg_side = np.sin(step_phase + np.pi) * 0.15
            pose[13, 0] += left_leg_side
            pose[14, 0] += right_leg_side
            pose[0:6, 0] += 0.2

        elif phase.name == "crouch_hop":
            hop_height = np.sin(progress * np.pi) * 0.2
            forward_motion = progress * 0.2
            crouch_amount = 0.3
            pose[6:10, 1] -= crouch_amount
            for joint_idx in range(15):
                pose[joint_idx, 1] += hop_height + forward_motion
            pose[13, 1] -= 0.2
            pose[14, 1] -= 0.2

        elif phase.name == "zigzag_hop":
            hop_height = np.sin(progress * np.pi * 2) * 0.3
            zigzag_x = np.sin(progress * np.pi * 4) * 0.2
            forward_y = progress * 0.3
            for joint_idx in range(15):
                pose[joint_idx, 0] += zigzag_x
                pose[joint_idx, 1] += hop_height + forward_y

        elif phase.name == "backward_hop":
            hop_height = np.sin(progress * np.pi) * 0.3
            backward_motion = progress * -0.25
            for joint_idx in range(15):
                pose[joint_idx, 1] += hop_height + backward_motion
            pose[6:10, 1] += 0.1

        elif phase.name == "side_turn_hop":
            hop_height = np.sin(progress * np.pi) * 0.35
            turn_angle = progress * np.pi * 0.5
            cos_a, sin_a = np.cos(turn_angle), np.sin(turn_angle)
            center_x, center_y = pose[7, 0], pose[7, 1]
            for joint_idx in range(15):
                rel_x = pose[joint_idx, 0] - center_x
                rel_y = pose[joint_idx, 1] - center_y
                pose[joint_idx, 0] = center_x + rel_x * cos_a - rel_y * sin_a
                pose[joint_idx, 1] = center_y + rel_x * sin_a + rel_y * cos_a + hop_height

        elif phase.name == "wing_tuck_walk":
            step_phase = (time * 2.5) % (2 * np.pi)
            forward_motion = progress * 0.3
            pose[:, 1] += forward_motion
            pose[10, 0] = pose[7, 0] - 0.03
            pose[10, 1] = pose[7, 1] - 0.05
            pose[11, 0] = pose[7, 0] + 0.03
            pose[11, 1] = pose[7, 1] - 0.05
            left_leg_phase = np.sin(step_phase)
            right_leg_phase = np.sin(step_phase + np.pi)
            pose[13, 0] += left_leg_phase * 0.08
            pose[14, 0] += right_leg_phase * 0.08

        elif phase.name == "static_folded_standing":
            body_sway = np.sin(time * 1.5) * 0.01
            pose[7:10, 0] += body_sway
            pose[10, 0] = pose[7, 0] - 0.03
            pose[10, 1] = pose[7, 1] - 0.02
            pose[11, 0] = pose[7, 0] + 0.03
            pose[11, 1] = pose[7, 1] - 0.02

        elif phase.name == "one_leg_folded_rest":
            pose[13, 1] += 0.2
            pose[10, 0] = pose[7, 0] - 0.03
            pose[10, 1] = pose[7, 1] - 0.02
            pose[11, 0] = pose[7, 0] + 0.03
            pose[11, 1] = pose[7, 1] - 0.02
            pose[6:10, 0] += 0.05

        elif phase.name == "folded_observing":
            observe_turn = np.sin(time * 1.0) * 0.2
            pose[0:6, 0] += observe_turn
            pose[10, 0] = pose[7, 0] - 0.03
            pose[10, 1] = pose[7, 1] - 0.02
            pose[11, 0] = pose[7, 0] + 0.03
            pose[11, 1] = pose[7, 1] - 0.02
            pose[6:10, 1] += 0.05

        elif phase.name == "relaxed_folded_stand":
            gentle_sway = np.sin(time * 0.8) * 0.005
            pose[7:10, 0] += gentle_sway
            pose[10, 0] = pose[7, 0] - 0.02
            pose[10, 1] = pose[7, 1] - 0.03
            pose[11, 0] = pose[7, 0] + 0.02
            pose[11, 1] = pose[7, 1] - 0.03

        elif phase.name == "walking":
            step_phase = (time * 3.0) % (2 * np.pi)
            forward_motion = progress * 0.5
            pose[:, 1] += forward_motion
            left_leg_phase = np.sin(step_phase)
            right_leg_phase = np.sin(step_phase + np.pi)
            pose[13, 0] += left_leg_phase * 0.1
            pose[13, 1] += abs(left_leg_phase) * 0.1
            pose[14, 0] += right_leg_phase * 0.1
            pose[14, 1] += abs(right_leg_phase) * 0.1
            body_bob = np.sin(step_phase * 2) * 0.05
            pose[6:10, 1] += body_bob

        elif phase.name == "pecking":
            peck_motion = np.sin(progress * np.pi * 2) * 0.4
            pose[0:6, 1] -= abs(peck_motion)
            pose[1, 1] -= peck_motion * 0.5
            pose[6, 1] -= peck_motion * 0.3

        elif phase.name == "preening":
            preen_phase = (time * 4.0) % (2 * np.pi)
            head_twist = np.sin(preen_phase) * 0.3
            pose[0:6, 0] += head_twist
            body_tilt = np.cos(preen_phase) * 0.1
            pose[6:10, 0] += body_tilt

    def _apply_aerial_behavior(self, pose, phase: BehaviorPhase, time, progress, species: BirdSpecies,
                               primary_phase, wing_amp, wind_effect_x, wind_effect_y):
        """Apply aerial behavior – enhanced large flapping"""

        wing_stroke = np.sin(primary_phase) * wing_amp * 3.0
        wing_sweep = np.cos(primary_phase * 1.3 + np.pi/4) * wing_amp * 0.7
        wing_twist = np.sin(primary_phase * 2.0) * 0.15

        cycle_progress = (primary_phase % (2 * np.pi)) / (2 * np.pi)
        if cycle_progress < 0.5:  # Downstroke
            power_stroke_mult = 1.8
            wing_angle_offset = -0.3
        else:                      # Upstroke
            power_stroke_mult = 0.5
            wing_angle_offset = 0.2

        wing_stroke *= power_stroke_mult

        if phase.name == "power_flap":
            wing_stroke *= 1.5
            wing_sweep *= 1.3
            body_power_bob = np.sin(primary_phase * 0.8) * 0.2
            pose[6:10, 1] += body_power_bob

        elif phase.name in ("glide", "soar"):
            wing_stroke *= 0.2
            wing_sweep *= 0.1
            pose[10, 0] = pose[7, 0] - species.wingspan * 0.8
            pose[11, 0] = pose[7, 0] + species.wingspan * 0.8
            pose[10, 1] = pose[7, 1] + 0.1
            pose[11, 1] = pose[7, 1] + 0.1

        elif "turn" in phase.name:
            direction = 1 if "right" in phase.name else -1
            turn_intensity = 1.5 if "sharp" in phase.name else 1.0
            inner_wing_idx = 11 if direction > 0 else 10
            outer_wing_idx = 10 if direction > 0 else 11
            pose[inner_wing_idx, 1] += direction * 0.15 * turn_intensity
            pose[outer_wing_idx, 1] -= direction * 0.1 * turn_intensity
            body_bank = direction * 0.2 * turn_intensity
            pose[6:10, 0] += body_bank
            pose[12, 0] += direction * 0.3 * turn_intensity

        elif phase.name == "hover":
            hover_stroke = np.sin(primary_phase * 2) * wing_amp * 0.8
            pose[10, 1] += hover_stroke
            pose[11, 1] += hover_stroke
            pose[6:10, 1] += 0.1

        elif phase.name == "barrel_roll":
            roll_phase = progress * 2 * np.pi
            for joint_idx in range(15):
                ox = pose[joint_idx, 0]
                oy = pose[joint_idx, 1]
                pose[joint_idx, 0] = ox * np.cos(roll_phase) - (oy - pose[7, 1]) * np.sin(roll_phase)
                pose[joint_idx, 1] = ox * np.sin(roll_phase) + (oy - pose[7, 1]) * np.cos(roll_phase) + pose[7, 1]

        if phase.name not in ["glide", "soar", "barrel_roll"]:
            pose[10, 0] += wing_sweep - wing_stroke * 0.3 + wind_effect_x
            pose[11, 0] -= wing_sweep - wing_stroke * 0.3 + wind_effect_x
            pose[10, 1] += wing_stroke * 2.5 + wing_twist + wing_angle_offset + wind_effect_y
            pose[11, 1] += wing_stroke * 2.5 - wing_twist * 0.9 + wing_angle_offset + wind_effect_y

    def _apply_transition_behavior(self, pose, phase: BehaviorPhase, time, progress, species: BirdSpecies,
                                   primary_phase, wing_amp):
        """Apply transition behavior"""

        if phase.name == "prepare_takeoff":
            crouch_amount = (1 - progress) * 0.3
            pose[7:10, 1] -= crouch_amount
            wing_prep = progress * 0.3
            pose[10, 0] -= wing_prep
            pose[11, 0] += wing_prep
            pose[13, 1] -= crouch_amount * 0.5
            pose[14, 1] -= crouch_amount * 0.5

        elif phase.name == "takeoff":
            takeoff_power = np.sin(progress * np.pi) * 2.0
            if progress < 0.3:
                leg_push = (0.3 - progress) / 0.3 * 0.5
                pose[13, 1] -= leg_push
                pose[14, 1] -= leg_push
            wing_stroke = np.sin(primary_phase) * wing_amp * 2.5
            pose[10, 1] += wing_stroke + takeoff_power * 0.3
            pose[11, 1] += wing_stroke + takeoff_power * 0.3
            body_angle = progress * 0.4
            pose[6:10, 1] += body_angle
            upward_motion = progress * 1.0
            pose[:, 1] += upward_motion

        elif phase.name == "landing_approach":
            approach_angle = progress * 0.3
            pose[6:10, 1] -= approach_angle
            brake_flap = np.sin(primary_phase) * wing_amp * 1.8
            pose[10, 1] += brake_flap
            pose[11, 1] += brake_flap
            leg_prep = progress * 0.4
            pose[13, 1] -= leg_prep
            pose[14, 1] -= leg_prep

        elif phase.name == "landing":
            landing_phase = progress * np.pi
            brake_stroke = np.sin(primary_phase) * wing_amp * 2.0
            pose[10, 1] += brake_stroke
            pose[11, 1] += brake_stroke
            leg_extend = np.sin(landing_phase) * 0.6
            pose[13, 1] -= leg_extend
            pose[14, 1] -= leg_extend
            body_settle = progress * 0.2
            pose[6:10, 1] -= body_settle
            pose[12, 1] += progress * 0.2

    def _apply_enhanced_biomechanical_constraints(self,
                                                  sequence: np.ndarray,
                                                  species: BirdSpecies) -> np.ndarray:
        """Enhanced biomechanical constraints"""

        T, J, C = sequence.shape
        constrained = sequence.copy()

        for t in range(T):
            pose = constrained[t]

            # 1) Reasonable spine order
            spine_joints = [1, 6, 7, 8, 9]
            for i in range(len(spine_joints)-1):
                curr_joint = spine_joints[i]
                next_joint = spine_joints[i+1]
                if pose[next_joint, 1] > pose[curr_joint, 1] + 0.2:
                    pose[next_joint, 1] = pose[curr_joint, 1] + 0.1

            # 2) Bone length constraints
            for (j1, j2), (min_len, max_len) in self.joint_constraints['bone_lengths'].items():
                curr_len = euclidean(pose[j1, :2], pose[j2, :2])
                target_len = min_len * species.body_length
                max_allowed = max_len * species.body_length
                if curr_len > max_allowed:
                    direction = pose[j2, :2] - pose[j1, :2]
                    direction = direction / curr_len * max_allowed
                    pose[j2, :2] = pose[j1, :2] + direction
                elif curr_len < target_len:
                    direction = pose[j2, :2] - pose[j1, :2]
                    direction = direction / curr_len * target_len
                    pose[j2, :2] = pose[j1, :2] + direction

            # 3) Ground constraint
            ground_level = -0.5 * species.body_length
            if pose[13, 1] < ground_level:
                pose[13, 1] = ground_level
            if pose[14, 1] < ground_level:
                pose[14, 1] = ground_level

            # 4) Wing angle constraints
            back_pos = pose[7, :2]
            left_wing_pos = pose[10, :2]
            right_wing_pos = pose[11, :2]
            left_angle = np.arctan2(left_wing_pos[1] - back_pos[1], left_wing_pos[0] - back_pos[0])
            right_angle = np.arctan2(right_wing_pos[1] - back_pos[1], right_wing_pos[0] - back_pos[0])
            min_angle, max_angle = self.joint_constraints['angle_limits']['wing_fold']

            if left_angle < min_angle:
                dist = euclidean(back_pos, left_wing_pos)
                pose[10, 0] = back_pos[0] + np.cos(min_angle) * dist
                pose[10, 1] = back_pos[1] + np.sin(min_angle) * dist

            if right_angle > -min_angle:
                dist = euclidean(back_pos, right_wing_pos)
                pose[11, 0] = back_pos[0] + np.cos(-min_angle) * dist
                pose[11, 1] = back_pos[1] + np.sin(-min_angle) * dist

        return constrained

    def _advanced_smoothing(self, sequence: np.ndarray) -> np.ndarray:
        """Advanced smoothing – preserve key motion characteristics"""

        T, J, C = sequence.shape
        smoothed = sequence.copy()

        for j in range(J):
            for c in range(2):  # smooth x,y
                if T > 5:
                    if j <= 5:      # head
                        window = min(7, T if T % 2 == 1 else T - 1); order = 3
                    elif j in [10, 11]:  # wings
                        window = min(5, T if T % 2 == 1 else T - 1); order = 2
                    elif j in [13, 14]:  # legs
                        window = min(3, T if T % 2 == 1 else T - 1); order = 1
                    else:            # torso
                        window = min(7, T if T % 2 == 1 else T - 1); order = 3
                    try:
                        smoothed[:, j, c] = savgol_filter(sequence[:, j, c],
                                                          window_length=window,
                                                          polyorder=order)
                    except:
                        pass
        return smoothed

    def _allocate_frames_intelligent(self, phases: List[BehaviorPhase],
                                     total_frames: int) -> List[int]:
        """Intelligent frame allocation based on behavior importance"""

        importance_weights = []
        for phase in phases:
            category_bonus = {'transition': 1.5, 'aerial': 1.2, 'ground': 1.0}
            weight = phase.energy_level * category_bonus.get(phase.category, 1.0)
            importance_weights.append(weight)

        total_weight = sum(importance_weights) if sum(importance_weights) > 0 else 1.0
        normalized_weights = [w / total_weight for w in importance_weights]

        allocations = []
        remaining_frames = total_frames

        for i, (phase, weight) in enumerate(zip(phases, normalized_weights)):
            if i == len(phases) - 1:
                allocations.append(remaining_frames)
            else:
                min_frames, max_frames = phase.duration_range
                ideal_frames = int(weight * total_frames)
                allocated = np.clip(ideal_frames, min_frames,
                                    min(max_frames, remaining_frames - (len(phases) - i - 1)))
                allocations.append(int(allocated))
                remaining_frames -= int(allocated)

        return allocations


def generate_diverse_enhanced_behavior_dataset(num_sequences: int = 1000,
                                               sequence_length: int = 64,
                                               include_environment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a diverse enhanced behavior dataset + primary label (six classes)"""

    def pick_primary_label(phases: List[BehaviorPhase]) -> int:
        # Accumulate score as (mean duration × category weight) for canonical classes (skip ground)
        score = {k: 0.0 for k in CANONICAL_LABELS}
        for ph in phases:
            canon = PHASE_TO_CANON.get(ph.name, None)
            if canon is None:
                continue
            dur = 0.5 * (ph.duration_range[0] + ph.duration_range[1])
            w_cat = 1.5 if ph.category == "transition" else (1.2 if ph.category == "aerial" else 1.0)
            score[canon] += dur * w_cat
        if all(v == 0 for v in score.values()):
            return LABEL_TO_ID["gliding"]
        primary = max(score.items(), key=lambda x: x[1])[0]
        return LABEL_TO_ID[primary]

    generator = EnhancedCUBBirdGenerator()
    all_sequences: List[np.ndarray] = []
    all_labels: List[int] = []

    print(f"Generating {num_sequences} CUB-15 enhanced behavior sequences...")
    print(f"Features: full behavior library + ground motions + large wing flaps + static folded poses + smart transitions")

    behavior_stats = {category: 0 for category in ['ground', 'aerial', 'transition', 'mixed']}

    for i in range(num_sequences):
        # Random species + individual variation
        species = np.random.choice(BIRD_SPECIES)
        species_variant = BirdSpecies(
            name=species.name,
            wingspan=species.wingspan * np.random.uniform(0.85, 1.15),
            body_length=species.body_length * np.random.uniform(0.9, 1.1),
            wing_loading=species.wing_loading * np.random.uniform(0.8, 1.2),
            flap_frequency=species.flap_frequency * np.random.uniform(0.7, 1.3),
            glide_ratio=species.glide_ratio * np.random.uniform(0.6, 1.0),
            muscle_power=species.muscle_power * np.random.uniform(0.8, 1.2),
            agility=species.agility * np.random.uniform(0.7, 1.3),
            hop_ability=species.hop_ability * np.random.uniform(0.8, 1.2),
            ground_preference=species.ground_preference * np.random.uniform(0.5, 1.5)
        )

        # Intelligent behavior combination
        behavior_combination, sequence_type = select_intelligent_behavior_combination(species_variant)
        behavior_stats[sequence_type] += 1

        # Environment
        environment = None
        if include_environment:
            environment = EnvironmentCondition(
                wind_speed=np.random.uniform(0, 3.0),
                wind_direction=np.random.uniform(0, 2*np.pi),
                turbulence=np.random.uniform(0, 0.3),
                temperature=np.random.uniform(10, 35)
            )

        # Generate sequence
        sequence = generator.generate_behavior_sequence(
            species_variant, behavior_combination, sequence_length, environment
        )
        all_sequences.append(sequence)

        # Primary label
        label_id = pick_primary_label(behavior_combination)
        all_labels.append(label_id)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{num_sequences} ({(i+1)/num_sequences*100:.1f}%)")

    dataset = np.array(all_sequences, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Stats
    print(f"\nBehavior type distribution:")
    for behavior_type, count in behavior_stats.items():
        percentage = count / num_sequences * 100
        print(f"  {behavior_type}: {count} ({percentage:.1f}%)")

    # Quality checks
    validate_enhanced_dataset_quality(dataset)

    return dataset, labels


def select_intelligent_behavior_combination(species: BirdSpecies) -> Tuple[List[BehaviorPhase], str]:
    """Intelligently select a behavior combination"""

    ground_weight = species.ground_preference * 2.0
    aerial_weight = (1.0 - species.ground_preference) * 2.0
    transition_weight = 1.0

    sequence_type_weights = {
        'ground': ground_weight,
        'aerial': aerial_weight,
        'transition': transition_weight,
        'mixed': 1.5
    }

    total_weight = sum(sequence_type_weights.values())
    rand = np.random.random() * total_weight
    cumsum = 0
    sequence_type = 'mixed'

    for seq_type, weight in sequence_type_weights.items():
        cumsum += weight
        if rand <= cumsum:
            sequence_type = seq_type
            break

    available_behaviors = {
        'ground': [b for b in BEHAVIOR_PHASES if b.category == 'ground'],
        'aerial': [b for b in BEHAVIOR_PHASES if b.category == 'aerial'],
        'transition': [b for b in BEHAVIOR_PHASES if b.category == 'transition']
    }

    selected_behaviors: List[BehaviorPhase] = []

    if sequence_type == 'ground':
        num_behaviors = np.random.randint(2, 5)
        selected_behaviors = np.random.choice(
            available_behaviors['ground'],
            size=min(num_behaviors, len(available_behaviors['ground'])),
            replace=False
        ).tolist()

    elif sequence_type == 'aerial':
        num_behaviors = np.random.randint(2, 4)
        selected_behaviors = np.random.choice(
            available_behaviors['aerial'],
            size=min(num_behaviors, len(available_behaviors['aerial'])),
            replace=False
        ).tolist()

    elif sequence_type == 'transition':
        selected_behaviors = np.random.choice(
            available_behaviors['transition'],
            size=min(3, len(available_behaviors['transition'])),
            replace=False
        ).tolist()

    else:  # mixed
        story_templates = [
            ['static_folded_standing', 'head_turn_left', 'prepare_takeoff', 'takeoff', 'cruise', 'landing'],
            ['hop_forward', 'pecking', 'hop_right', 'relaxed_folded_stand'],
            ['prepare_takeoff', 'takeoff', 'climb', 'turn_left', 'glide', 'landing_approach', 'landing'],
            ['one_leg_folded_rest', 'walking', 'hop_left', 'preening'],
            ['takeoff', 'power_flap', 'sharp_turn_right', 'dive', 'landing'],
            ['static_folded_standing', 'head_turn_right', 'hop_forward', 'prepare_takeoff', 'takeoff', 'hover'],
            ['folded_observing', 'side_hop_left', 'wing_tuck_walk', 'static_folded_standing'],
            ['relaxed_folded_stand', 'folded_wing_hop', 'diagonal_hop', 'one_leg_folded_rest'],
            ['static_folded_standing', 'side_walk', 'crouch_hop', 'folded_observing']
        ]
        template = story_templates[np.random.randint(len(story_templates))]
        name_to_phase = {phase.name: phase for phase in BEHAVIOR_PHASES}
        selected_behaviors = [name_to_phase[name] for name in template if name in name_to_phase]

    if not selected_behaviors:
        selected_behaviors = [np.random.choice(BEHAVIOR_PHASES)]

    return selected_behaviors, sequence_type


def validate_enhanced_dataset_quality(dataset: np.ndarray) -> None:
    """Validate enhanced dataset quality"""

    print(f"\nEnhanced dataset quality check:")
    N, T, J, C = dataset.shape

    print(f"  Shape: {dataset.shape}")
    print(f"  Data range: [{dataset.min():.3f}, {dataset.max():.3f}]")

    motion_std = np.std(dataset[:, :, :, :2], axis=1)
    avg_motion = np.mean(motion_std)
    motion_variety = np.std(np.mean(motion_std, axis=(1, 2)))

    print(f"  Mean motion amplitude: {avg_motion:.3f}")
    print(f"  Motion diversity: {motion_variety:.3f}")

    height_changes = []
    for i in range(min(100, N)):
        seq_heights = dataset[i, :, 9, 1]  # belly joint height
        height_range = seq_heights.max() - seq_heights.min()
        height_changes.append(height_range)

    avg_height_change = np.mean(height_changes)
    print(f"  Mean altitude change: {avg_height_change:.3f}")

    wing_activities = []
    for i in range(min(100, N)):
        left_wing_activity = np.std(dataset[i, :, 10, :2])
        right_wing_activity = np.std(dataset[i, :, 11, :2])
        wing_activities.append((left_wing_activity + right_wing_activity) / 2)

    avg_wing_activity = np.mean(wing_activities)
    print(f"  Mean wing activity: {avg_wing_activity:.3f}")

    wing_y_ranges = []
    for i in range(min(50, N)):
        left_wing_y_range = dataset[i, :, 10, 1].max() - dataset[i, :, 10, 1].min()
        right_wing_y_range = dataset[i, :, 11, 1].max() - dataset[i, :, 11, 1].min()
        wing_y_ranges.append((left_wing_y_range + right_wing_y_range) / 2)

    avg_wing_y_range = np.mean(wing_y_ranges)
    print(f"  Mean wing Y-range: {avg_wing_y_range:.3f}")

    if avg_motion > 0.5 and motion_variety > 0.1:
        print("  OK: good motion diversity")
    else:
        print("  Warning: insufficient motion diversity")

    if avg_height_change > 0.3:
        print("  OK: flight behaviors present")
    else:
        print("  Warning: lacks prominent flight behaviors")

    if avg_wing_activity > 0.2:
        print("  OK: active wing motion")
    else:
        print("  Warning: insufficient wing motion")

    if avg_wing_y_range > 0.8:
        print("  OK: strong large-amplitude wing flaps")
    else:
        print("  Warning: wing flap amplitude is small")


# Data augmentation
def create_augmented_dataset(base_dataset: np.ndarray,
                             augmentation_factor: int = 3) -> np.ndarray:
    """Create an augmented dataset (returns augmented data; labels should be duplicated externally)"""

    print(f"Creating augmented dataset (factor: {augmentation_factor})...")

    N, T, J, C = base_dataset.shape
    augmented_sequences = [base_dataset]

    for aug_round in range(augmentation_factor - 1):
        print(f"  Augmentation round {aug_round + 1}/{augmentation_factor - 1}")
        augmented_batch = np.zeros_like(base_dataset)

        for i in range(N):
            sequence = base_dataset[i].copy()

            # Apply augmentations
            if np.random.random() < 0.4:
                speed_factor = np.random.uniform(0.8, 1.25)
                sequence = time_warp_sequence(sequence, speed_factor)

            if np.random.random() < 0.5:
                scale_factor = np.random.uniform(0.9, 1.1)
                rotation_angle = np.random.uniform(-0.1, 0.1)
                sequence = spatial_transform_sequence(sequence, scale_factor, rotation_angle)

            if np.random.random() < 0.3:
                sequence = mirror_sequence(sequence)

            if np.random.random() < 0.3:
                noise_level = np.random.uniform(0.01, 0.03)
                sequence = add_controlled_noise(sequence, noise_level)

            augmented_batch[i] = sequence

        augmented_sequences.append(augmented_batch)

    final_dataset = np.concatenate(augmented_sequences, axis=0)
    print(f"  Done: {base_dataset.shape[0]} -> {final_dataset.shape[0]}")
    return final_dataset


def time_warp_sequence(sequence: np.ndarray, speed_factor: float) -> np.ndarray:
    """Time warping augmentation"""
    T, J, C = sequence.shape
    warped = np.zeros_like(sequence)

    original_times = np.arange(T)
    warped_times = np.linspace(0, T-1, T) * speed_factor
    warped_times = np.clip(warped_times, 0, T-1)

    for j in range(J):
        for c in range(C):
            warped[:, j, c] = np.interp(warped_times, original_times, sequence[:, j, c])

    return warped


def spatial_transform_sequence(sequence: np.ndarray,
                               scale_factor: float,
                               rotation_angle: float) -> np.ndarray:
    """Spatial transform augmentation"""
    transformed = sequence.copy()

    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)

    for t in range(len(sequence)):
        transformed[t, :, 0] *= scale_factor
        transformed[t, :, 1] *= scale_factor

        x = transformed[t, :, 0].copy()
        y = transformed[t, :, 1].copy()
        transformed[t, :, 0] = x * cos_a - y * sin_a
        transformed[t, :, 1] = x * sin_a + y * cos_a

    return transformed


def mirror_sequence(sequence: np.ndarray) -> np.ndarray:
    """Mirror augmentation"""
    mirrored = sequence.copy()
    mirrored[:, :, 0] *= -1

    # Swap left/right symmetric joints
    left_right_pairs = [(3, 4), (10, 11), (13, 14)]
    for left_idx, right_idx in left_right_pairs:
        temp = mirrored[:, left_idx, :].copy()
        mirrored[:, left_idx, :] = mirrored[:, right_idx, :]
        mirrored[:, right_idx, :] = temp

    return mirrored


def add_controlled_noise(sequence: np.ndarray, noise_level: float) -> np.ndarray:
    """Controlled noise augmentation"""
    noisy = sequence.copy()

    joint_noise_factors = {
        'head': 0.5, 'body': 0.7, 'wings': 1.2, 'tail': 0.8, 'legs': 0.6
    }

    head_noise = np.random.normal(0, noise_level * joint_noise_factors['head'],
                                  (len(sequence), 6, 2))
    noisy[:, :6, :2] += head_noise

    body_noise = np.random.normal(0, noise_level * joint_noise_factors['body'],
                                  (len(sequence), 4, 2))
    noisy[:, 6:10, :2] += body_noise

    wing_noise = np.random.normal(0, noise_level * joint_noise_factors['wings'],
                                  (len(sequence), 2, 2))
    noisy[:, 10:12, :2] += wing_noise

    tail_noise = np.random.normal(0, noise_level * joint_noise_factors['tail'],
                                  (len(sequence), 1, 2))
    noisy[:, 12:13, :2] += tail_noise

    leg_noise = np.random.normal(0, noise_level * joint_noise_factors['legs'],
                                 (len(sequence), 2, 2))
    noisy[:, 13:15, :2] += leg_noise

    return noisy


# Main
if __name__ == "__main__":
    print("Enhanced CUB-15 Behavior Generator (Large Wing-Flap + Training Labels)")
    print("=" * 70)

    # 1) Generate base dataset + labels
    print("\n[1] Generating enhanced behavior dataset...")
    base_dataset, base_labels = generate_diverse_enhanced_behavior_dataset(
        num_sequences=1000,
        sequence_length=64,
        include_environment=True
    )

    # Save base dataset and labels
    np.save("cub15_enhanced_base_dataset_large_wing.npy", base_dataset)
    np.save("cub15_enhanced_base_labels_large_wing.npy", base_labels)
    print(f"Saved: cub15_enhanced_base_dataset_large_wing.npy / cub15_enhanced_base_labels_large_wing.npy")

    # 2) Create augmented dataset (duplicate labels externally by the same factor)
    print("\n[2] Creating augmented dataset...")
    augmentation_factor = 4
    augmented_dataset = create_augmented_dataset(base_dataset, augmentation_factor=augmentation_factor)

    # Align labels with augmented data
    full_labels = np.concatenate([base_labels.copy() for _ in range(augmentation_factor)], axis=0)
    np.save("cub15_enhanced_full_dataset_large_wing.npy", augmented_dataset)
    np.save("cub15_enhanced_full_labels_large_wing.npy", full_labels)
    print(f"Saved: cub15_enhanced_full_dataset_large_wing.npy / cub15_enhanced_full_labels_large_wing.npy")

    # 3) Train/val split (stratified)
    print("\n[3] Train / Val split...")
    from sklearn.model_selection import train_test_split

    train_data, val_data, train_labels, val_labels = train_test_split(
        augmented_dataset, full_labels, test_size=0.1, random_state=42, stratify=full_labels
    )

    np.save("cub15_enhanced_train_pose_large_wing.npy", train_data)
    np.save("cub15_enhanced_val_pose_large_wing.npy", val_data)
    np.save("cub15_enhanced_train_labels.npy", train_labels)
    np.save("cub15_enhanced_val_labels.npy", val_labels)

    # Stats
    print(f"\nFinal dataset stats:")
    print(f"  Train set: {train_data.shape}, labels: {train_labels.shape}")
    print(f"  Val set:   {val_data.shape}, labels: {val_labels.shape}")
    print(f"  Total sequences: {len(augmented_dataset):,}")

    print(f"\nDone. Enhanced CUB-15 dataset generated (large wing-flap version + labels).")
    print(f"Generated files:")
    print(f"  - cub15_enhanced_base_dataset_large_wing.npy (base enhanced data)")
    print(f"  - cub15_enhanced_base_labels_large_wing.npy (base labels)")
    print(f"  - cub15_enhanced_full_dataset_large_wing.npy (augmented data)")
    print(f"  - cub15_enhanced_full_labels_large_wing.npy (augmented labels)")
    print(f"  - cub15_enhanced_train_pose_large_wing.npy (train set)")
    print(f"  - cub15_enhanced_val_pose_large_wing.npy (val set)")
    print(f"  - cub15_enhanced_train_labels.npy (train labels)")
    print(f"  - cub15_enhanced_val_labels.npy (val labels)")
    print(f"\nWing flaps now have large amplitudes, include more static folded-wing poses, and each sequence carries a primary label among six classes.")
