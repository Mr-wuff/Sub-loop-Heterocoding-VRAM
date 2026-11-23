"""Pure R62 gate helpers: public baseline evidence, NOT an SHV novelty test."""
from __future__ import annotations
import statistics


def compare(tiled: dict, liger: dict, p: dict) -> dict:
    if tiled.get('status') != 'OK' or liger.get('status') != 'OK':
        raise ValueError('Only complete OK cases can be compared')
    fields = ('initial_parameter_sha256', 'input_sha256', 'parameter_count',
              'checkpoint_enabled', 'resident_state_bytes')
    same = all(tiled.get(k) == liger.get(k) for k in fields)
    same &= all(tiled.get('environment', {}).get(k) == liger.get('environment', {}).get(k)
                for k in ('gpu', 'cuda', 'torch', 'transformers', 'bitsandbytes', 'liger_kernel'))
    same &= bool(tiled.get('checkpoint_enabled'))
    q_t = tiled.get('quantization_audit', {})
    q_l = liger.get('quantization_audit', {})
    q_ok = all(q.get('large_active_parameters', 0) > 0 and
               q['large_active_parameters'] == q.get('quantized_uint8_moment_pairs') and
               q['large_active_parameters'] == q.get('blockwise_absmax_pairs') for q in (q_t, q_l))
    xt, xl = tiled['steps'], liger['steps']
    if len(xt) != len(xl):
        raise ValueError('Mismatched step count')
    def rel(a, b):
        return abs(a - b) / max(abs(a), 1e-12)
    loss = max(rel(a['loss'], b['loss']) for a, b in zip(xt, xl))
    probe = rel(tiled['final_probe_loss'], liger['final_probe_loss'])
    g = p['gate']
    quality = loss <= g['relative_loss_trajectory_max'] and probe <= g['relative_probe_max']
    memory_saving = 1.0 - tiled['peak_mib'] / liger['peak_mib']
    time_ratio = tiled['median_step_ms'] / liger['median_step_ms']
    valid = same and q_ok and quality
    return {
        'same_environment_and_inputs': bool(same),
        'real_eightbit': bool(q_ok), 'quality_pass': bool(quality), 'valid': bool(valid),
        'relative_loss_max': loss, 'relative_probe_gap': probe,
        'tiled_vs_liger_memory_saving': memory_saving,
        'tiled_vs_liger_time_ratio': time_ratio,
        'tiled_strictly_better_in_this_run': memory_saving > 0 and time_ratio < 1.0,
        'material_gain_pass': memory_saving >= g['material_peak_gain_min'] and time_ratio <= g['acceptable_tiled_time_ratio_max'],
        'fast_tiled': time_ratio <= 1.0 - g['stable_speed_improvement_min']
    }


def aggregate(comparisons: list[dict], p: dict) -> dict:
    if len(comparisons) != len(p['repeat_order']) or not all(c['valid'] for c in comparisons):
        return {'status':'R62_INCOMPLETE_OR_PARITY_HOLD'}
    mem = [c['tiled_vs_liger_memory_saving'] for c in comparisons]
    speed = [c['tiled_vs_liger_time_ratio'] for c in comparisons]
    passing = sum(c['material_gain_pass'] for c in comparisons)
    fast = sum(c['fast_tiled'] for c in comparisons)
    enough = p['gate']['stable_repeat_count_min']
    # No 'Pareto tradeoff' label for a tiny numerical win with no material gain.
    status = ('R62_PUBLIC_MATERIAL_GAIN_REPLICATION_SIGNAL_NO_ORIGINALITY_CLAIM'
              if passing >= enough else 'R62_PUBLIC_BASELINE_ENVELOPE_NO_MATERIAL_SHV_GAIN')
    return {
        'status': status,
        'replicates': len(comparisons),
        'material_gain_repeats': passing, 'tiled_faster_at_least_10pct_repeats': fast,
        'tiled_speed_signal_replicates': fast >= enough,
        'median_memory_saving': statistics.median(mem),
        'min_memory_saving': min(mem), 'max_memory_saving': max(mem),
        'median_time_ratio': statistics.median(speed),
        'min_time_ratio': min(speed), 'max_time_ratio': max(speed),
        'no_shv_originality_claim': True
    }
