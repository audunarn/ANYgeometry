"""UNWIRED exact-containment index experiment; root alone executes/replays.

No geometric certificate, admission decision or production index is changed.
``replay(witnesses)`` accepts an already saved witness sequence, with no archive
path dependency. A previous region containing a new witness's owned cell would
identify a redundant local proof at its creation, not prove a new curve result.
"""
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest


def _box(value):
    result = tuple(tuple(Q(x) for x in axis) for axis in value)
    if len(result) != 4 or any(len(axis) != 2 or not 0 <= axis[0] <= axis[1] <= 1 for axis in result):
        raise ValueError("expected four exact span-local intervals in [0,1]")
    return result


def _bucket(value):
    scaled = value*16
    return min(15,max(0,scaled.numerator//scaled.denominator))


def _contains(region, cell):
    return all(a <= lo <= hi <= b for (lo,hi),(a,b) in zip(cell,region))


class CoverageIndexPrototype:
    """Four independent 16-bin bitset filters, followed by exact containment.

    Each span pair has LOCAL bit positions: sparse global witness IDs cannot
    inflate every span's masks. Both endpoints of all four cell axes are needed.
    Every surviving bit is examined; there is no prefix truncation. Live storage
    is O(W), queries are O(W) in the worst case, and mask operations scale with
    the local witness bit count. This is a filter, not a proof authority service.
    """

    def __init__(self, tick=lambda:None):
        self.tick = tick
        self.records = []
        self.spans = {}
        self.exact_checks = 0
        self.mask_checks = 0
        self.last_checks = 0

    def insert(self, spans, region):
        self.tick()
        region = _box(region)
        index = len(self.records)
        if spans not in self.spans:
            self.spans[spans] = ([],[[0]*16 for _ in range(4)])
        ids,masks = self.spans[spans]
        bit = 1 << len(ids)
        # All work that can invoke cancellation happens before publishing the
        # record/ID/masks, so an interrupted insertion cannot leave a half entry.
        updates = []
        for axis,(lo,hi) in enumerate(region):
            self.tick()
            for bucket in range(_bucket(lo),_bucket(hi)+1):
                updates.append((axis,bucket,masks[axis][bucket] | bit))
        self.records.append((spans,region))
        ids.append(index)
        for axis,bucket,value in updates:
            masks[axis][bucket] = value
        return index

    def find(self, spans, cell):
        """First insertion-order containing ID, or None; poll all candidate work."""
        self.tick()
        cell = _box(cell)
        self.last_checks = 0
        entry = self.spans.get(spans)
        if entry is None:
            return None
        ids,masks = entry
        candidates = None
        for axis,(lo,hi) in enumerate(cell):
            self.tick()
            necessary = masks[axis][_bucket(lo)] & masks[axis][_bucket(hi)]
            self.mask_checks += 2
            candidates = necessary if candidates is None else candidates & necessary
            if not candidates:
                return None
        while candidates:
            self.tick()
            bit = candidates & -candidates
            local = bit.bit_length()-1
            index = ids[local]
            self.exact_checks += 1
            self.last_checks += 1
            if _contains(self.records[index][1],cell):
                return index
            candidates ^= bit
        return None


def replay(witnesses, tick=lambda:None):
    """Offline creation-order replay; no kernel run or geometric validation.

    The exhaustive baseline tests EVERY earlier same-span witness. Max bin
    occupancy reports the legacy one-axis buckets, not the new bitset cost.
    A zero real-miss count does not establish this mechanism as a bottleneck.
    """
    index = CoverageIndexPrototype(tick)
    legacy,earlier = {},{}
    report = dict(witnesses=0,legacy_hits=0,complete_hits=0,exhaustive_hits=0,
                  missed_earlier_containments=0,avoidable_new_certificates=0,
                  legacy_exact_checks=0,complete_exact_checks=0,exhaustive_exact_checks=0,
                  max_legacy_bin_occupancy=0,complete_mask_checks=0,missed_witness_ids=[],
                  new_filter_misses=0)
    for wi,witness in enumerate(witnesses):
        tick()
        spans,cell,region = witness.spans,_box(witness.cell),_box(witness.region)
        old = None
        for old_id,old_region in legacy.get(spans,{}).get(_bucket(cell[0][0]),())[:64]:
            tick()
            report["legacy_exact_checks"] += 1
            if _contains(old_region,cell):
                old = old_id
                break
        brute = None
        for old_id,old_region in earlier.get(spans,()):
            tick()
            report["exhaustive_exact_checks"] += 1
            if _contains(old_region,cell):
                brute = old_id
                break
        found = index.find(spans,cell)
        report["legacy_hits"] += int(old is not None)
        report["complete_hits"] += int(found is not None)
        report["exhaustive_hits"] += int(brute is not None)
        if found != brute:
            report["new_filter_misses"] += 1
            raise AssertionError("complete filter differs from exhaustive first containment")
        if old is None and brute is not None:
            report["missed_earlier_containments"] += 1
            report["missed_witness_ids"].append(wi)
        report["avoidable_new_certificates"] += int(brute is not None)
        assert index.insert(spans,region) == wi
        earlier.setdefault(spans,[]).append((wi,region))
        bins = legacy.setdefault(spans,{})
        for bucket in range(_bucket(region[0][0]),_bucket(region[0][1])+1):
            bins.setdefault(bucket,[]).append((wi,region))
            report["max_legacy_bin_occupancy"] = max(report["max_legacy_bin_occupancy"],len(bins[bucket]))
        report["witnesses"] += 1
    report["complete_exact_checks"] = index.exact_checks
    report["complete_mask_checks"] = index.mask_checks
    report["missed_witness_ids"] = tuple(report["missed_witness_ids"])
    return report


SPAN = ((0,0),(0,0))
UNIT = ((Q(0),Q(1)),)*4


def test_legacy_prefix_starves_later_valid_proof():
    decoy = ((Q(0),Q(1)),(Q(0),Q(1,4)),(Q(0),Q(1)),(Q(0),Q(1)))
    good = ((Q(0),Q(1)),(Q(1,2),Q(1)),(Q(0),Q(1)),(Q(0),Q(1)))
    cell = ((Q(1,3),Q(1,2)),(Q(3,4),Q(7,8)),(Q(1,3),Q(1,2)),(Q(1,3),Q(1,2)))
    witnesses = [SimpleNamespace(spans=SPAN,cell=decoy,region=decoy) for _ in range(64)]
    witnesses += [SimpleNamespace(spans=SPAN,cell=good,region=good),
                  SimpleNamespace(spans=SPAN,cell=cell,region=cell)]
    report = replay(witnesses)
    assert 65 in report["missed_witness_ids"]
    assert report["missed_earlier_containments"] == 1 and report["new_filter_misses"] == 0
    assert report["max_legacy_bin_occupancy"] == 66


def test_boundaries_point_cells_and_span_isolation():
    index = CoverageIndexPrototype()
    other = ((1,0),(0,0))
    assert index.insert(SPAN,UNIT) == 0
    for value in (Q(0),Q(1,16),Q(1,2),Q(1)):
        assert index.find(SPAN,((value,value),)*4) == 0
    assert index.find(other,UNIT) is None
    assert index.insert(other,UNIT) == 1
    assert index.find(other,UNIT) == 1
    assert len(index.spans[other][0]) == 1
    assert all(mask==1 for axis in index.spans[other][1] for mask in axis)


def test_filter_agrees_with_exhaustive_fraction_boxes():
    index = CoverageIndexPrototype()
    for i in range(48):
        box = tuple((Q((i*(k+1))%12,16),Q((i*(k+1))%12+4,16)) for k in range(4))
        index.insert(SPAN if i%2 else ((2,0),(0,0)),box)
    for i in range(96):
        cell = tuple((Q((i*(k+3))%15,16),Q((i*(k+3))%15+1,16)) for k in range(4))
        spans = SPAN if i%3 else ((2,0),(0,0))
        expected = next((j for j,(s,region) in enumerate(index.records) if s==spans and _contains(region,cell)),None)
        assert index.find(spans,cell) == expected


def test_cancellation_during_insert_and_candidate_checks():
    state = {"armed":False,"calls":0,"stop":3}
    def tick():
        if state["armed"]:
            state["calls"] += 1
            if state["calls"] == state["stop"]:
                raise RuntimeError("cancelled prototype")
    index = CoverageIndexPrototype(tick)
    index.insert(SPAN,UNIT)
    state["armed"] = True
    with pytest.raises(RuntimeError,match="cancelled"):
        index.insert(SPAN,UNIT)
    assert len(index.records) == len(index.spans[SPAN][0]) == 1
    state["calls"] = 0
    state["stop"] = 6  # entry + four axis polls, then the candidate itself
    with pytest.raises(RuntimeError,match="cancelled"):
        index.find(SPAN,UNIT)


def test_no_matches_remains_an_index_answer_only():
    index=CoverageIndexPrototype()
    assert index.find(SPAN,UNIT) is None
    with pytest.raises(ValueError):
        index.insert(SPAN,((Q(-1),Q(1)),)*4)
